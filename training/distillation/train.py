import gc
import datetime
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ultralytics import YOLO
from utils import logger
from training.data.preprocess.tf.embedding_preprocessor import EmbeddingPreprocessor
from training.data.preprocess.torch.detector_preprocess import detector_preprocessing
from training.data.data_loaders.torch.deck_dataset import DecklistDataset
from training.loss.torch.detection_loss import v8DetectionLoss
from models.embedding_model import EmbeddingModel
from models.detector import OneStageDetector
from training.configs.mapper.distillation import load_distillation_config


def make_adamw(model, lr=1e-4, momentum=0.9, decay=0.01):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    for param_name, param in model.named_parameters():
        if "bias" in param_name:  # bias (no decay)
            g[2].append(param)
        elif isinstance(param, bn):  # weight (no decay)
            g[1].append(param)
        else:  # weight (width decay)
            g[0].append(param)

    optimizer = torch.optim.AdamW(
        g[2],
        lr=lr,
        betas=(momentum, 0.999),
        weight_decay=0.0,
    )
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

    return optimizer


def make_dataset(df_path, deck_size, image_dir):
    X_train = pd.read_csv(df_path)
    id_list = X_train["id"].tolist()
    id_list = [str(Path(image_dir) / f"{card_id}.jpg") for card_id in id_list]
    card_type = list(
        map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
    )
    return DecklistDataset(id_list, card_type, deck_shape=deck_size)


def run_one_epoch(
    epoch,
    dataloader,
    student_model,
    teacher_model,
    loss_fn,
    is_train,
    optimizer,
    embedding_topk_start,
    embedding_topk_interval,
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
            img_width, img_height, _ = img.shape # deck recipe image

            # get indivisual card position
            xyxy_list = batch["xyxy"][mask.numpy()]
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
        if is_train:
            preds = student_model(student_inputs)
        else:
            with torch.no_grad():
                preds = student_model(student_inputs)

        # loss
        embed_topk = embedding_topk_start + epoch // embedding_topk_interval
        loss, loss_item, fg_mask = loss_fn(preds, batch, embed_topk=embed_topk)
        total_loss += loss_item.detach().cpu()
        sample_count += batch_size

        # back propagation
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # norm_loss = grad_norm(loss_item[[0, 1, 3]])
            # print(grad_norm.weights)
            optimizer.step()

    return total_loss / sample_count


def main():
    config = load_distillation_config()
    use_logger = True
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log = logger.TrainLogger() if use_logger else None

    # prepare datasets
    train_dataset = DecklistDataset.load_from_csv(
        str(config.data.train_csv),
        config.data.deck_size_range,
        image_dir=str(config.data.card_image_dir),
    )
    train_loader = DataLoader(
        train_dataset,
        config.training.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    # valid_dataset = make_dataset("datasets/valid.csv", 1)
    # valid_loader = DataLoader(
    #     valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn
    # )

    # prepare student model
    pretrained_model = YOLO(str(config.model.student.pretrained_weights))
    pre_model_dict = pretrained_model.model.model.state_dict()
    student_model = OneStageDetector()
    model_dict = student_model.state_dict()

    for key, value in pre_model_dict.items():
        source_prefix = key.split(".", 1)[0]
        target_key = f"layer{key}" if source_prefix.isdigit() else key
        if (
            "embedding" not in target_key
            and target_key in model_dict
            and model_dict[target_key].shape == value.shape
        ):
            model_dict[target_key] = value.detach().clone()
    student_model.load_state_dict(model_dict)
    del pretrained_model, pre_model_dict

    for name, param in student_model.named_parameters():
        if config.model.student.train_embedding_only:
            if "embedding_layers" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            param.requires_grad_(False if "dfl" in name else True)

    student_model = student_model.to(device)

    # prevent TF model occupies all memories
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            raise e

    # prepare teacher model
    teacher_model = EmbeddingModel()
    teacher_model.load(str(config.model.teacher.embedding_weights))

    # losses
    # box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
    det_loss = v8DetectionLoss(
        head=student_model.layer22,
        device=device,
        box_gain=config.training.loss.box_gain,
        cls_gain=config.training.loss.class_gain,
        dfl_gain=config.training.loss.dfl_gain,
        embed_gain=config.training.loss.embedding_gain,
    )
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
    )

    # grad norm
    # grad_norm = gradNorm.GradNorm(3, student_model.layer21)

    # training
    best_loss = torch.inf
    checkpoint_dir = config.checkpoint.directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.training.epochs):
        train_dataset.shuffle()
        train_loss = run_one_epoch(
            epoch,
            train_loader,
            student_model,
            teacher_model,
            det_loss,
            True,
            optimizer,
            config.training.embedding_topk.start,
            config.training.embedding_topk.increase_every_epochs,
        )

        if use_logger:
            log.info(f"epoch: {epoch}")
            log.info("train loss")
            log.info(
                f"det_loss: {train_loss[0]} cls_loss: {train_loss[1]} dfl_loss: {train_loss[2]} embed_loss: {train_loss[3]}"
            )

        gc.collect()
        torch.cuda.empty_cache()


        gc.collect()
        torch.cuda.empty_cache()

        loss = train_loss.sum().detach().cpu().item()
        if loss < best_loss:
            torch.save(
                student_model.state_dict(),
                checkpoint_dir / config.checkpoint.save_best_as,
            )
            best_loss = loss
        torch.save(
            student_model.state_dict(), checkpoint_dir / config.checkpoint.save_last_as
        )


if __name__ == "__main__":
    main()
