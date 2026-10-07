from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from training.configs.mapper.common import OptimizerConfig, StrictConfig, load_yaml_config


class DataConfig(StrictConfig):
    train_csv: Path
    valid_csv: Path
    card_image_dir: Path


class ModelConfig(StrictConfig):
    input_shape: tuple[int, int, int]


class AugmentationConfig(StrictConfig):
    min_ratio: float = Field(ge=0, le=1)
    max_ratio: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_ratio_order(self):
        if self.min_ratio >= self.max_ratio:
            raise ValueError("min_ratio must be less than max_ratio")
        return self


class SchedulerConfig(StrictConfig):
    type: Literal["cosine", "exponential"]
    warmup_epochs: int = Field(ge=0)
    total_epochs: int = Field(gt=0)
    decay_rate: float = Field(gt=0, le=1)
    staircase: bool


class ContrastiveLossConfig(StrictConfig):
    margin: float = Field(gt=0)


class TrainingConfig(StrictConfig):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    validation_batch_size: int = Field(gt=0)
    hard_select_start_epoch: int = Field(ge=0)
    matrix_refresh_interval: int = Field(gt=0)
    augmentation: AugmentationConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    contrastive_loss: ContrastiveLossConfig


class CheckpointConfig(StrictConfig):
    directory: Path
    resume_from: str | None = None
    save_best_as: str
    save_last_as: str


class EmbeddingConfig(StrictConfig):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig

    @model_validator(mode="after")
    def validate_scheduler_epochs(self):
        if self.training.scheduler.total_epochs != self.training.epochs:
            raise ValueError("scheduler.total_epochs must match training.epochs")
        return self


def load_embedding_config(
    path: str | Path = "training/configs/embedding.yaml",
) -> EmbeddingConfig:
    return load_yaml_config(path, EmbeddingConfig)