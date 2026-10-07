import torch
import torch.nn as nn
from ultralytics.utils.loss import DFLoss, bbox_iou, bbox2dist
from ultralytics.utils.ops import xywh2xyxy
from training.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .distillation_loss import kd_loss, rkd_loss


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(
        self,
        head,
        device,
        tal_topk=10,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        embed_gain=1,
    ):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""

        m = head  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = {
            "box": box_gain,
            "cls": cls_gain,
            "dfl": dfl_gain,
            "embed": embed_gain,
        }
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.embedding_size = m.embedding_size
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def preprocess_embedding(self, targets, batch_size):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, embed_topk=3):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        pred_detect = preds[0]
        pred_embeds = preds[1]
        feats = pred_detect[1] if isinstance(pred_detect, tuple) else pred_detect
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # embeeding
        pred_embeds = torch.cat(
            [xi.view(feats[0].shape[0], self.embedding_size, -1) for xi in pred_embeds],
            dim=2,
        )
        pred_embeds = pred_embeds.permute(0, 2, 1).contiguous()
        l2_norm = pred_embeds.norm(2, dim=-1, keepdim=True)
        pred_embeds = pred_embeds / l2_norm

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        targets_embed = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["embedding"]), dim=1
        )
        gt_embeds = self.preprocess_embedding(targets_embed.to(self.device), batch_size)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        del targets, targets_embed

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, target_embeds, fg_mask, target_gt_idx = (
            self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                (pred_embeds.detach()).type(gt_embeds.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                gt_embeds,
                mask_gt,
            )
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2], iou_scores = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        iou_cls_scores = (iou_scores * target_scores).sum(-1)  # (b, w*h)
        embed_mask = torch.zeros_like(fg_mask).bool()
        for batch_index in range(target_gt_idx.shape[0]):
            for instance_id in torch.unique(target_gt_idx[batch_index]):
                instance_mask = target_gt_idx[batch_index] == instance_id
                topk_val, topk_idx = torch.topk(
                    iou_cls_scores[batch_index] * instance_mask, embed_topk
                )
                embed_mask[batch_index][topk_idx] = True

        # Embedding loss (euclidean distance)
        # embed_weight = iou_cls_scores[embed_mask]
        kd = kd_loss(pred_embeds[embed_mask], target_embeds[embed_mask])
        rkd = rkd_loss(target_embeds[embed_mask], pred_embeds[embed_mask])
        # loss[3] = (kd + rkd) * embed_weight / embed_weight.sum()
        loss[3] = kd + rkd

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp["dfl"]  # dfl gain
        loss[3] *= self.hyp["embed"]

        # loss(box, cls, dfl)
        return loss.sum(), loss.detach().cpu(), fg_mask


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True)
        iou_scores = (1.0 - iou[fg_mask]) * weight
        loss_iou = iou_scores.sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou
