from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODEL_EXTS = {".pt", ".pth", ".onnx"}


@dataclass
class LoadedModel:
    path: Path
    kind: str


def resolve_model_path(model_dir: str, model_name: Optional[str] = None) -> Optional[Path]:
    """Resolve a model path from a directory and optional name."""
    base = Path(model_dir)
    if not base.exists():
        return None
    if model_name:
        candidate = base / model_name
        return candidate if candidate.exists() else None

    candidates = []
    for ext in MODEL_EXTS:
        candidates.extend(base.rglob(f"*{ext}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_model(model_path: Optional[Path]) -> Optional[LoadedModel]:
    """Create a lightweight model descriptor from a path."""
    if model_path is None:
        return None
    suffix = model_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        kind = "pytorch"
    elif suffix == ".onnx":
        kind = "onnx"
    else:
        kind = "unknown"
    return LoadedModel(path=model_path, kind=kind)
