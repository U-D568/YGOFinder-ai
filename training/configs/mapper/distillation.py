from pathlib import Path

from pydantic import Field

from training.configs.mapper.common import OptimizerConfig, StrictConfig, load_yaml_config


class DataConfig(StrictConfig):
	train_csv: Path
	valid_csv: Path
	card_image_dir: Path
	deck_size_range: tuple[int, int]


class StudentModelConfig(StrictConfig):
	pretrained_weights: Path
	train_embedding_only: bool


class TeacherModelConfig(StrictConfig):
	embedding_weights: Path


class ModelConfig(StrictConfig):
	student: StudentModelConfig
	teacher: TeacherModelConfig


class LossConfig(StrictConfig):
	box_gain: float = Field(ge=0)
	class_gain: float = Field(ge=0)
	dfl_gain: float = Field(ge=0)
	embedding_gain: float = Field(ge=0)


class EmbeddingTopKConfig(StrictConfig):
	start: int = Field(gt=0)
	increase_every_epochs: int = Field(gt=0)


class TrainingConfig(StrictConfig):
	epochs: int = Field(gt=0)
	batch_size: int = Field(gt=0)
	optimizer: OptimizerConfig
	loss: LossConfig
	embedding_topk: EmbeddingTopKConfig


class CheckpointConfig(StrictConfig):
	directory: Path
	save_best_as: str
	save_last_as: str


class DistillationConfig(StrictConfig):
	data: DataConfig
	model: ModelConfig
	training: TrainingConfig
	checkpoint: CheckpointConfig


def load_distillation_config(
	path: str | Path = "training/configs/distillation.yaml",
) -> DistillationConfig:
	return load_yaml_config(path, DistillationConfig)
