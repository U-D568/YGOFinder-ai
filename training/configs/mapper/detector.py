from pathlib import Path

from pydantic import Field

from training.configs.mapper.common import StrictConfig, load_yaml_config


class ModelConfig(StrictConfig):
    pretrained_weights: Path


class DataConfig(StrictConfig):
    dataset_yaml: Path


class TrainingConfig(StrictConfig):
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)


class PredictionConfig(StrictConfig):
    image: Path


class DetectorConfig(StrictConfig):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    prediction: PredictionConfig


def load_detector_config(
    path: str | Path = "training/configs/detector.yaml",
) -> DetectorConfig:
    return load_yaml_config(path, DetectorConfig)
