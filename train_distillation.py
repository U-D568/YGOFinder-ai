"""Train an isolated student candidate while preserving the teacher embedding space."""
import argparse
import csv
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from data.preprocess.tf import EmbeddingPreprocessor
from data.preprocess.torch import detector_preprocessing
from data.dataset.torch import DecklistDataset
from loss.torch.detection_loss import v8DetectionLoss
from models.tf import EmbeddingModel
from models.torch import Detector


def run_one_epoch(
    epoch,
    dataloader,
    student_model,
    teacher_model,
    loss_fn,
    optimizer,
):
    teacher_preprocess = EmbeddingPreprocessor()
    device = next(student_model.parameters()).device
    dtype = next(student_model.parameters()).dtype
    total_loss = torch.zeros(4)
    sample_count = 0

    for batch in dataloader:
        # ground-truth preprocess
        images = batch["image"]
        batch_size = images.shape[0]
        batch["bboxes"] = torch.from_numpy(batch["xywh"])
        batch["batch_idx"] = torch.from_numpy(batch["batch_idx"])
        batch["cls"] = torch.zeros(size=batch["batch_idx"].shape, dtype=dtype)

        # make ground-truth embedding
        gt_embeds = []
        for i, img in enumerate(images):
            mask = batch["batch_idx"] == i
            teacher_inputs = []
            img_height, img_width, _ = img.shape # deck recipe image

            # get indivisual card position
            xyxy_list = batch["xyxy"][mask.numpy()].copy()
            xyxy_list[:, [0, 2]] *= img_width
            xyxy_list[:, [1, 3]] *= img_height
            xyxy_list = np.round(xyxy_list).astype(np.int32)

            # crop card image
            for xyxy in xyxy_list:
                x1, y1, x2, y2 = xyxy
                crop = img[y1:y2, x1:x2, :]
                crop = teacher_preprocess.resize(crop)
                teacher_inputs.append(crop)
            teacher_inputs = tf.stack(teacher_inputs)
            teacher_embeds = teacher_model(teacher_inputs).numpy()
            teacher_embeds = torch.from_numpy(teacher_embeds)
            gt_embeds.append(teacher_embeds)
        batch["embedding"] = torch.concatenate(gt_embeds, dim=0)

        # student preprocess
        student_inputs = detector_preprocessing(images).to(device)

        # inference
        student_model.train()
        preds = student_model(student_inputs)

        # loss
        embed_topk = epoch // 50 + 1
        loss, loss_item, _ = loss_fn(preds, batch, embed_topk=embed_topk)
        total_loss += loss_item.detach().cpu()
        sample_count += batch_size

        # back propagation
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()

    return total_loss / sample_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--train-csv", default="datasets/train.csv")
    parser.add_argument("--image-dir", default="datasets/card_images_small")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if output.exists() or output.resolve() == Path(args.checkpoint).resolve():
        parser.error("output must be a new candidate path")
    with open(args.train_csv, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training data is empty")
    for row in rows:
        if not row.get("type"):
            raise ValueError("Training CSV requires id and type")
        image = Path(args.image_dir) / (row["id"] + ".jpg")
        if not image.is_file():
            raise FileNotFoundError(image)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DecklistDataset.load_from_csv(
        args.train_csv, (1, 4), image_dir=str(Path(args.image_dir)) + "/"
    )
    if not len(dataset):
        raise ValueError("Training data is empty")
    loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    student = Detector()
    # Resume all layers, including the embedding head that is used by Chroma search.
    student.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    for name, parameter in student.named_parameters():
        parameter.requires_grad_("dfl" not in name)
    student.to(device)
    teacher = EmbeddingModel(model_path=args.teacher)
    teacher.model.trainable = False
    loss_fn = v8DetectionLoss(head=student.layer22, device=device, embed_gain=1.0)
    optimizer = AdamW((p for p in student.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        dataset.shuffle()
        losses = run_one_epoch(epoch, loader, student, teacher, loss_fn, optimizer)
        print(f"epoch={epoch + 1} losses={losses.tolist()}", flush=True)
    # Candidate selection is done by held-out recognition evaluation, not training loss.
    torch.save(student.state_dict(), output)


if __name__ == "__main__":
    main()
