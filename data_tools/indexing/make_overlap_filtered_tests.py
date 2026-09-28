# -*- coding: utf-8 -*-
"""Write test sets with the tiles that overlap training removed.

The HisMap split is a checkerboard on a 384-px grid, and the last row and column of each sheet are
shifted back to the sheet edge, so some test tiles share most of their pixels with a training tile
(293 of 9,486 above 50%, 513 above 10%). This writes, per subset:

    test_samples_ov<p>.txt                               SMOL's target list (--target-file)
    sample_pair_<sel>_test_samples_ov<p>_gap0.json       cd_syntra2's pairing, same targets

where <p> is the overlap threshold in percent. A test tile is dropped when its pixel overlap with
any training tile of the same subset exceeds p%. Subsets whose tile names carry no grid position
(Siegfried railway and vineyard) are kept whole, and the counts say so.

    python data_tools/indexing/make_overlap_filtered_tests.py -d /home/yuan/data/syntra \\
        --selections cls_selection_100,budget_random_gap0_1000t_s2026 --thresholds 10,50
"""

import argparse
import glob
import json
import os

TILE = 384


def grid(names):
    try:
        return [tuple(int(v) for v in n.rsplit(".", 1)[1].split("_")) for n in names]
    except ValueError:
        return None


def overlap(a, b):
    return max(0, TILE - abs(a[0] - b[0])) * max(0, TILE - abs(a[1] - b[1])) / TILE / TILE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--root-dir", required=True)
    p.add_argument("--selections", required=True, help="comma-separated selection names")
    p.add_argument("--thresholds", default="10,50", help="overlap thresholds in percent")
    args = p.parse_args()

    dirs = sorted(glob.glob(f"{args.root_dir}/hismap/*")) + \
        [f"{args.root_dir}/siegfried/railway", f"{args.root_dir}/siegfried/vineyard"]
    sels = args.selections.split(",")
    for t in (int(x) for x in args.thresholds.split(",")):
        dropped = kept = unchecked = 0
        for d in dirs:
            test = open(f"{d}/test_samples.txt").read().split()
            train = open(f"{d}/train_samples.txt").read().split()
            tg, rg = grid(test), grid(train)
            if tg is None or rg is None:
                keep, unchecked = test, unchecked + len(test)
            else:
                keep = [n for n, a in zip(test, tg)
                        if max((overlap(a, b) for b in rg), default=0.0) <= t / 100]
            dropped += len(test) - len(keep)
            kept += len(keep)
            with open(f"{d}/test_samples_ov{t}.txt", "w") as f:
                f.write("\n".join(keep) + "\n")
            keep = set(keep)
            for sel in sels:
                src = f"{d}/sample_pair_{sel}_test_samples_gap0.json"
                with open(src) as f:
                    pairing = json.load(f)
                with open(f"{d}/sample_pair_{sel}_test_samples_ov{t}_gap0.json", "w") as f:
                    json.dump({k: v for k, v in pairing.items() if k in keep}, f)
        print(f"> {t}% overlap: dropped {dropped} test tiles, kept {kept} "
              f"({unchecked} of them in subsets without grid names, not checked)")


if __name__ == "__main__":
    main()
