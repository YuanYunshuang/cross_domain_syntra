# -*- coding: utf-8 -*-
"""Rewrite a tile-based selection as the (tile, class) passes it consumes, balanced.

The n-shot scheme picks tiles and the release is exhaustively labelled, so training took every
class of every selected tile - 4,859 tiles became 17,753 samples, 10,912 of them negative. Under
a pass budget that is not free: checking a tile for water and finding none is a pass.

Every positive is kept, and negatives are sampled to match their number rather than taken
wholesale, so the arm carries the balanced composition the budgeted arm is built to. The 61.5%
negative skew of the original was an accident of exhaustive labelling, not a design choice, and
matching it would spend three passes on absence for every two on presence.

    PYTHONPATH=. python data_tools/indexing/expand_selection.py \\
        -d /home/yuan/data/syntra --source cls_selection_100 --name e13x1_nshot \\
        --output-root /home/yuan/data/syntra_e13 -dkw hismap,siegfried -xkw hameln.3921_2023
"""

import argparse
import json
import os

import numpy as np

from data_tools.indexing.build_global_index import subset_dirs
from data_tools.indexing.make_pool_pairing import fg_threshold, mirror_subset

INVALID_CLS = ("background", "nonlabeled")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--source", required=True, help="selection to expand, without .json")
    parser.add_argument("--name", required=True, help="output name")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--fg-ratio", default="0.05")
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="negatives kept per positive. 0 writes the positive half alone, "
                             "for when the negatives are shared with another arm")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="hold this share out as validation, as select_budget does, so the "
                             "two arms train on the same number of passes")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))
    rng = np.random.RandomState(args.seed)

    dirs = subset_dirs(args.root_dir, args.data_keywords, args.exclude_keys)
    total_tiles = n_pos = n_neg = n_val = 0
    for subset_dir in dirs:
        subset = os.path.basename(subset_dir)
        path = os.path.join(subset_dir, f"{args.source}.json")
        if not os.path.isfile(path):
            print(f"  !! {subset}: no {args.source}.json")
            continue
        with open(os.path.join(subset_dir, "color_map.json")) as f:
            classes = sorted(c for c in json.load(f) if c not in INVALID_CLS)
        with open(os.path.join(subset_dir, f"{subset}.json")) as f:
            ratios = json.load(f)
        tiles = sorted({n for names in json.load(open(path)).values() for n in names})
        threshold = fg_threshold(subset, fg_ratio)

        positives, negatives = [], []
        for tile in tiles:
            for cls in classes:
                target = positives if ratios.get(tile, {}).get(cls, 0.0) > threshold else negatives
                target.append((tile, cls))
        want = int(round(args.negative_ratio * len(positives)))
        keep = rng.choice(len(negatives), min(want, len(negatives)), replace=False)
        chosen = positives + [negatives[int(i)] for i in sorted(keep)]

        # the same stratified holdout select_budget applies, so neither arm trains on more
        val_pairs = []
        if args.val_fraction:
            per_class = {}
            for pair in chosen:
                per_class.setdefault(pair[1], []).append(pair)
            for cls, group in sorted(per_class.items()):
                take = int(round(args.val_fraction * len(group)))
                if 0 < take < len(group):
                    idx = rng.choice(len(group), take, replace=False)
                    val_pairs += [group[int(i)] for i in idx]
            val_set = set(val_pairs)
            chosen = [p for p in chosen if p not in val_set]

        for suffix, pairs in (("", chosen), ("_val", val_pairs)):
            if suffix and not val_pairs:
                continue
            by_class = {}
            for tile, cls in pairs:
                by_class.setdefault(cls, []).append(tile)
            out_dir = subset_dir
            if args.output_root:
                out_dir = mirror_subset(subset_dir, args.output_root, subset, args.root_dir)
            with open(os.path.join(out_dir, f"{args.name}{suffix}.json"), "w") as f:
                json.dump({c: sorted(v) for c, v in sorted(by_class.items())}, f, indent=2)
        total_tiles += len(tiles)
        n_pos += len(positives)
        n_neg += len(keep)
        n_val += len(val_pairs)
    print(f"{args.name}: {total_tiles} tiles -> {n_pos + n_neg} passes "
          f"({n_pos} positive, {n_neg} negative)"
          + (f", {n_val} held out for validation" if n_val else ""))


if __name__ == "__main__":
    main()
