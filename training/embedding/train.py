import os
import sys
import time
import math
from datetime import datetime
import gc

import tensorflow as tf
from tensorflow.keras.optimizers import AdamW

sys.path.append(f"{os.getcwd()}")

from training.loss.tf.loss import contrastive_loss
from utils import logger
from training.data.data_loaders.tf.embedding_dataset import EmbeddingDataset
from training.data.augmentation.tf.embeds import EmbeddingAugmentation
from structures.embedding_matrix import EmbeddingMatrix
from models.embedding_model import EmbeddingModel

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        raise e

class TFWarmUpScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, base_lr, warmup_epochs, total_epochs, steps_per_epoch, decay_type="cosine"
    ):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.decay_type = decay_type

    def __call__(self, step):
        # 현재 epoch 및 step 계산
        epoch = step // self.steps_per_epoch
        global_step = tf.cast(step, tf.float32)

        # warmup 단계
        if epoch < self.warmup_epochs:
            warmup_steps = self.warmup_epochs * self.steps_per_epoch
            return self.base_lr * (global_step / warmup_steps)

        # decay 단계
        decay_steps = (self.total_epochs - self.warmup_epochs) * self.steps_per_epoch
        decay_step = global_step - self.warmup_epochs * self.steps_per_epoch

        if self.decay_type == "cosine":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.base_lr, decay_steps=decay_steps
            )(decay_step)
        elif self.decay_type == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.base_lr,
                decay_steps=decay_steps,
                decay_rate=0.96,
                staircase=True,
            )(decay_step)
        else:
            return self.base_lr  # fallback to constant lr


def main():
    log = logger.TrainLogger()

    # train hyper parameter
    EPOCHS = 20
    HARD_SELECT = 0
    BATCH_SIZE = 16
    INPUT_SHAPE = (224, 224, 3)
    SAVE_PATH = "training/embedding/weights"

    # preprocessor
    augmentation = EmbeddingAugmentation(min_ratio=0.1, max_ratio=0.4)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    train_start = time.time()

    model = EmbeddingModel(INPUT_SHAPE)

    # data preparation
    train_dataset = EmbeddingDataset.load("training/datasets/train.csv")
    # valid_dataset = EmbeddingDataset.load("datasets/valid.csv")

    resume_path = os.path.join(SAVE_PATH, "best.h5")
    if os.path.isfile(resume_path):
        model.load(resume_path)
    total_steps = math.ceil(len(train_dataset) / BATCH_SIZE)
    warmup = TFWarmUpScheduler(
        base_lr=1e-4, warmup_epochs=3, total_epochs=EPOCHS, steps_per_epoch=total_steps
    )
    optimizer = AdamW(learning_rate=warmup, weight_decay=1e-4)

    train_matrix = EmbeddingMatrix(model, train_dataset)

    gc.collect()

    # training
    best_loss = float("inf")
    for epoch in range(100):
        epoch_start = time.time()
        train_loss = 0

        if epoch >= HARD_SELECT and epoch % 3 == 0:
            train_matrix.update_matrix()

        for batch in train_dataset.dataset.batch(BATCH_SIZE):
            anchor_img, indices = batch
            batch_size = anchor_img.shape[0]
            positive_img = augmentation(anchor_img)
            pred_positive = model(anchor_img)

            # select negative data
            negative_index = []
            if epoch >= HARD_SELECT:
                for pos, idx in zip(pred_positive, indices):
                    negative_index.append(train_matrix.get_hard_negative(pos, idx))
            else:
                for idx in indices:
                    negative_index.append(train_matrix.get_random_negative(idx))

            negative_img = []
            for index in negative_index:
                img = train_dataset[index]
                negative_img.append(img)
            negative_img = tf.stack(negative_img, axis=0)
            negative_img = augmentation(negative_img)

            with tf.GradientTape() as tape:
                pred_anchor = model(anchor_img)
                pred_positive = model(positive_img)
                pred_negative = model(negative_img)
                pos_loss = contrastive_loss(pred_anchor, pred_positive, 0, 0.2)
                neg_loss = contrastive_loss(pred_anchor, pred_negative, 1, 0.2)
                loss = tf.reduce_mean(pos_loss + neg_loss)
            gradients = tape.gradient(loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

            train_loss += loss.numpy() * batch_size

        gc.collect()
        # logging
        epoch_time = time.time() - epoch_start
        log.info(f"epoch: {epoch} {datetime.now().strftime('%Y-%m-%dT %H:%M:%S')}")
        log.info(f"\ttrain loss: {train_loss / len(train_dataset)}")
        log.info(f"\tprocessing time: {int(epoch_time) // 60}m {epoch_time % 60:.3f}s")

        # save best only
        train_loss
        if train_loss < best_loss:
            model.save(os.path.join(SAVE_PATH, "best.h5"))
            best_loss = train_loss
        model.save(os.path.join(SAVE_PATH, "last.h5"))
        gc.collect()

    total_time = time.time() - train_start
    log.info(f"total time: {int(total_time) // 60}m {total_time % 60:.3f}s")


if __name__ == "__main__":
    main()
