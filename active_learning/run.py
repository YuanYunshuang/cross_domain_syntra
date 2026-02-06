from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from active_learning.gui.app import run_gui
from active_learning.utils.config import (
    ActiveLearningConfig,
    build_config,
    load_yaml_config,
    merge_config,
)
from active_learning.utils.data import generate_next_round_dataset
from active_learning.utils.model import load_model, resolve_model_path


def _coerce_int(value: Any, key: str) -> Any:
    """Coerce a value to int if possible, preserving None."""
    if isinstance(value, int) or value is None:
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc
    return value


def _normalize_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config field types in-place and return the dict."""
    for key in ("round", "query_size", "seed", "tile_size"):
        config[key] = _coerce_int(config.get(key), key)
    return config


def _default_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill default model/output paths and expand user paths."""
    repo_root = Path(__file__).resolve().parents[1]
    if not config.get("model_dir"):
        config["model_dir"] = str(repo_root / "syntra")
    if not config.get("output_dir"):
        config["output_dir"] = str(repo_root / "active_learning" / "outputs")
    for key in ("data_dir", "model_dir", "output_dir"):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser())
    return config


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the active learning runner."""
    parser = argparse.ArgumentParser(description="Active learning runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--use-gui", action="store_true", help="Launch GUI to override settings")
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--query-size", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    return parser.parse_args()


def _args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed args into a config mapping."""
    return {
        "project_name": args.project_name,
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "round": args.round,
        "query_size": args.query_size,
        "strategy": args.strategy,
        "seed": args.seed,
        "device": args.device,
        "model_name": args.model_name,
        "tile_size": args.tile_size,
    }


def resolve_config(args: argparse.Namespace) -> ActiveLearningConfig:
    """Resolve config with YAML < CLI < GUI precedence."""
    config: Dict[str, Any] = {}
    if args.config:
        config = load_yaml_config(args.config)

    cli_overrides = _args_to_dict(args)
    config = merge_config(config, cli_overrides)
    config = _default_paths(config)

    gui_requested = bool(args.use_gui or config.get("use_gui"))
    if gui_requested:
        gui_overrides = run_gui(config) or {}
        gui_overrides = {
            key: value for key, value in gui_overrides.items() if value != ""
        }
        config = merge_config(config, gui_overrides)

    config = _normalize_config_types(config)
    return build_config(config)


def run_pipeline(config: ActiveLearningConfig) -> None:
    """Run the active learning pipeline for a single round."""
    model_path = resolve_model_path(config.model_dir, config.model_name)
    model = load_model(model_path)

    selected = generate_next_round_dataset(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        round_index=config.round,
        query_size=config.query_size,
        seed=config.seed,
        strategy=config.strategy,
    )

    print("Active Learning Run")
    print(f"Project: {config.project_name}")
    print(f"Model: {model.path if model else 'None'}")
    print(f"Selected {len(selected)} samples -> {Path(config.output_dir) / f'round_{config.round}' / 'selected.txt'}")


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    config = resolve_config(args)
    if not config.data_dir:
        raise ValueError("data_dir is required (set in YAML, CLI, or GUI).")
    run_pipeline(config)


if __name__ == "__main__":
    main()
