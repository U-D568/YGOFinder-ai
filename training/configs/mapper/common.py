from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerConfig(StrictConfig):
    name: Literal["AdamW"] = "AdamW"
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml_config(path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config_type.model_validate(raw_config)