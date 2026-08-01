# -*- coding: utf-8 -*-
"""Collect the evaluation results of all runs into one table.

Walks a log root, parses every ``test_*.txt`` written by ``syntra/test.py`` and emits one CSV
row per (run, evaluation, class) plus the overall mean of each evaluation. The run settings
that distinguish experiments (model, dataset class, k, sampler, data selection) are read from
the config block that ``test.py`` logs at the top of the file, so runs stay comparable even
when their configs have since been edited.

Examples:
    PYTHONPATH=. python syntra/tools/collect_results.py logs /koko/SYNTRA_logs \\
        -o results/all_runs.csv

    # compact overview on stdout, one line per evaluation
    PYTHONPATH=. python syntra/tools/collect_results.py /koko/SYNTRA_logs --summary
"""

import argparse
import ast
import csv
import os
import re
from glob import glob

# settings worth keeping next to every number; extend as experiments add switches
CONFIG_FIELDS = [
    "model_name", "dataset_name", "backbone", "dora_rank", "prompt_encoder", "mask_decoder",
    "num_src", "num_notions", "img_size", "src_selection", "src_fusion", "retrieval_encoder",
    "data_info", "data_keywords", "sampler", "batch_size", "accum_steps", "lr",
    "max_iterations", "ce_neg_weight", "dice_pos_weight", "dice_neg_weight", "root_dir",
    "checkpoint",
]
METRICS = ["iou", "precision", "recall", "accuracy"]

_METRICS_PATTERN = (
    r"\s*IoU \(FG\):\s*([\d.]+)"
    r"\s*Precision \(FG\):\s*([\d.]+)"
    r"\s*Recall \(FG\):\s*([\d.]+)"
    r"\s*Accuracy \(BG&FG\):\s*([\d.]+)"
)
_NOTION_RE = re.compile(r"Notion set : (\S+?):" + _METRICS_PATTERN)
_MEAN_RE = re.compile(r"Overal Mean #+" + _METRICS_PATTERN)
_LOSS_RE = re.compile(r"Average Loss:\s*([\d.]+)")


def parse_config_block(text):
    """Extract the effective test config that test.py logs as ``_content: {...}``."""
    match = re.search(r"_content: (\{.*?\})\s*\nINFO", text, re.DOTALL)
    if match is None:
        match = re.search(r"_content: (\{.*\})", text)
    if match is None:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return {}


def parse_test_file(path):
    """Parse one ``test_*.txt`` into a list of evaluation blocks, oldest first.

    ``test.py`` opens its log in append mode, so one file can hold several evaluations of the
    same run (different checkpoints, resumed runs, repeated calls). Each of them starts with a
    fresh config dump, which is what we split on.
    """
    with open(path, "r", errors="replace") as f:
        text = f.read()

    marker = "INFO:SYNTRA:_metadata:"
    chunks = [marker + c for c in text.split(marker)[1:]] or [text]

    blocks = []
    for chunk in chunks:
        classes = []
        for m in _NOTION_RE.finditer(chunk):
            name, iou, prec, rec, acc = m.groups()
            classes.append({
                "class": name.strip(),
                "iou": float(iou), "precision": float(prec),
                "recall": float(rec), "accuracy": float(acc),
            })

        mean = None
        m = _MEAN_RE.search(chunk)
        if m:
            iou, prec, rec, acc = m.groups()
            mean = {"iou": float(iou), "precision": float(prec),
                    "recall": float(rec), "accuracy": float(acc)}

        if not classes and mean is None:
            continue
        loss = _LOSS_RE.search(chunk)
        blocks.append({
            "config": parse_config_block(chunk),
            "classes": classes,
            "mean": mean,
            "loss": float(loss.group(1)) if loss else None,
        })
    return blocks


def collect(log_roots, all_blocks=False):
    """Parse every test_*.txt under the given roots. Returns a list of CSV-ready rows.

    By default only the most recent evaluation of each file is kept; ``all_blocks`` keeps the
    earlier ones too, numbered in the ``block`` column.
    """
    rows = []
    seen = set()
    for root in log_roots:
        for path in sorted(glob(os.path.join(root, "**", "test_*.txt"), recursive=True)):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)

            blocks = parse_test_file(path)
            if not blocks:
                print(f"  (empty) {path}")
                continue
            kept = list(enumerate(blocks)) if all_blocks else [(len(blocks) - 1, blocks[-1])]

            run = os.path.basename(os.path.dirname(path))
            name = os.path.splitext(os.path.basename(path))[0]
            evaluation = name[len("test_"):] if name.startswith("test_") else name
            for index, block in kept:
                cfg = block["config"]
                base = {
                    "run": run,
                    "eval": evaluation,
                    "block": index,
                    "n_blocks": len(blocks),
                    "path": path,
                    "avg_loss": block["loss"],
                    **{k: cfg.get(k) for k in CONFIG_FIELDS},
                }
                for cls in block["classes"]:
                    rows.append({**base, "scope": "class", **cls})
                if block["mean"] is not None:
                    rows.append({**base, "scope": "mean", "class": "ALL", **block["mean"]})
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = (["run", "eval", "block", "n_blocks", "scope", "class"] + METRICS +
                  ["avg_loss"] + CONFIG_FIELDS + ["path"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """One line per evaluation, sorted by mIoU."""
    means = [r for r in rows if r["scope"] == "mean"]
    means.sort(key=lambda r: r["iou"], reverse=True)
    n_cls = {}
    for r in rows:
        if r["scope"] == "class":
            key = (r["path"], r["block"])
            n_cls[key] = n_cls.get(key, 0) + 1

    header = f"{'mIoU':>6}  {'#cls':>4}  {'k':>2}  {'run':<44} {'eval':<28} {'settings'}"
    print(header)
    print("-" * len(header))
    for r in means:
        settings = " ".join(filter(None, [
            r.get("dataset_name"), r.get("backbone"),
            f"n{r['num_notions']}" if r.get("num_notions") else None,
            r.get("sampler"),
            (r.get("data_info") or "").replace("sample_pair_", "").replace("_intracls", ""),
        ]))
        print(f"{r['iou']:6.2f}  {n_cls.get((r['path'], r['block']), 0):4d}  "
              f"{str(r.get('num_src') or '-'):>2}  {r['run'][:44]:<44} {r['eval'][:28]:<28} {settings}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_roots", nargs="+", help="one or more directories holding run folders")
    parser.add_argument("-o", "--output", type=str, default="results/all_runs.csv",
                        help="path of the CSV to write")
    parser.add_argument("--summary", action="store_true", help="also print a ranked overview")
    parser.add_argument("--all-blocks", action="store_true",
                        help="keep earlier evaluations of an appended log file, not only the last")
    args = parser.parse_args()

    rows = collect(args.log_roots, all_blocks=args.all_blocks)
    write_csv(rows, args.output)
    n_evals = len({(r["run"], r["eval"]) for r in rows})
    print(f"Parsed {n_evals} evaluations ({len(rows)} rows) -> {args.output}")
    if args.summary:
        print()
        print_summary(rows)


if __name__ == "__main__":
    main()
