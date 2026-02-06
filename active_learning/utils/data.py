from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_files(data_dir: Path, exts: Iterable[str] = IMAGE_EXTS) -> List[Path]:
    """Collect files under data_dir with the given extensions."""
    files: List[Path] = []
    for ext in exts:
        files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(set(files))


def select_for_labeling(
    pool_files: Sequence[Path],
    query_size: int,
    seed: int = 42,
    strategy: str = "random",
) -> List[Path]:
    """Select a subset of files for labeling using the chosen strategy."""
    if not pool_files:
        return []
    if strategy != "random":
        raise ValueError(f"Unsupported strategy: {strategy}")
    rng = random.Random(seed)
    if query_size >= len(pool_files):
        return list(pool_files)
    return rng.sample(list(pool_files), query_size)


def generate_next_round_dataset(
    data_dir: str,
    output_dir: str,
    round_index: int,
    query_size: int,
    seed: int = 42,
    strategy: str = "random",
) -> List[Path]:
    """Generate a selection list for the next round and write selected.txt."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    pool_files = _collect_files(data_path)
    selected = select_for_labeling(
        pool_files=pool_files,
        query_size=query_size,
        seed=seed,
        strategy=strategy,
    )

    round_dir = Path(output_dir) / f"round_{round_index}"
    round_dir.mkdir(parents=True, exist_ok=True)
    selection_file = round_dir / "selected.txt"
    with selection_file.open("w", encoding="utf-8") as f:
        for path in selected:
            f.write(f"{path}\n")
    return selected
