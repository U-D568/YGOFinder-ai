# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import DFL

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), embedding_size=256):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.embedding_size = embedding_size
        self.nl = len(ch)  # number of detection layers
        self.reg_max = (
            16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        )
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(
            ch[0], min(self.nc, 100)
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.embedding_layers = nn.ModuleList(
            nn.Sequential(
                Conv(x, self.embedding_size // 2, 3),
                Conv(self.embedding_size // 2, self.embedding_size, 3),
                Conv(self.embedding_size, self.embedding_size, 1),
                nn.Conv2d(self.embedding_size, self.embedding_size, 1)
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        embeds = []
        for i in range(self.nl):
            embeds.append(self.embedding_layers[i](x[i]))
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x, embeds
        y, cat_embeds = self._inference(x, embeds)
        return (y, cat_embeds)

    def _inference(self, x, embed):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        embed = torch.cat([xi.view(shape[0], self.embedding_size, -1) for xi in embed], 2)
        l2_norm = embed.norm(2, dim=1, keepdim=True)
        embed = embed / l2_norm
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in {
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        }:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = (
                self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))
                * self.strides
            )

        return torch.cat((dbox, cls.sigmoid()), 1), embed

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, predictions = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(max_det)
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat(
            [boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()],
            dim=-1,
        )


class ResidualConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
        super().__init__()
        self.residual_block = nn.Sequential(
            Conv(in_ch, mid_ch, kernel_size),
            Conv(mid_ch, out_ch, kernel_size),
        )
        self.projection = Conv(in_ch, out_ch, kernel_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.projection(x) + self.residual_block(x))

