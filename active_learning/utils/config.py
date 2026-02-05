from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ActiveLearningConfig:
    project_name: str = "active_learning_run"
    data_dir: str = ""
    model_dir: str = ""
    output_dir: str = ""
    round: int = 1
    query_size: int = 50
    strategy: str = "random"
    seed: int = 42
    device: str = "cpu"
    model_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        extra = base.pop("extra", {}) or {}
        base.update(extra)
        return base


def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at the top level.")
    return data


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def build_config(data: Dict[str, Any]) -> ActiveLearningConfig:
    known_keys = {
        "project_name",
        "data_dir",
        "model_dir",
        "output_dir",
        "round",
        "query_size",
        "strategy",
        "seed",
        "device",
        "model_name",
    }
    kwargs: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for key, value in data.items():
        if key in known_keys:
            kwargs[key] = value
        else:
            extra[key] = value
    return ActiveLearningConfig(**kwargs, extra=extra)
