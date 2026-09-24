# -*- coding: utf-8 -*-
"""Write nested, class-balanced prompt pools for inference-only adaptation sweeps.

A pool is the set of annotated tiles a practitioner would have for a sheet: N tiles drawn from
one subset's training split, round-robin over classes so that each class the sheet carries is
represented as early as possible, and nested so that the 10-tile pool is a prefix of the 50 and
the 50 of the 100. This is the scheme the E5 pools (`e5_pool_<n>.json`) follow.

The pool may come from a *different* subset than the one it prompts. That is what E7b needs:
the held-out 1996 survey prompted from the same footprint's 1898, 1974, 1982, 2005 or 2017
tiles, to ask whether accuracy decays with the temporal gap or whether survey style dominates.
The pool file is written into the target subset's folder, so `make_pool_pairing.py` keys its
candidates to the target's scope; the loader resolves each source tile to its own folder.

    PYTHONPATH=. python data_tools/indexing/make_eval_pools.py -d /home/yuan/data/syntra \\
        --pool-subset hameln.3922_2005 --into-subset hameln.3922_1996 \\
        --prefix e7b_pool_2005 --sizes 10,50,100

then pair it, restricting the targets to the held-out subset with a target list only it has:

    PYTHONPATH=. python data_tools/indexing/make_pool_pairing.py -d /home/yuan/data/syntra \\
        --index /home/yuan/data/syntra/global_index.npz -dkw hameln.3922 \\
        --source e7b_pool_2005_10 --target e7b_targets
"""

import argparse
import json
import os
import random

from data_tools.indexing.build_global_index import subset_dirs

INVALID_CLS = ("background", "nonlabeled")


def find_subset(root, subset):
    matches = [d for d in subset_dirs(root, subset) if os.path.basename(d) == subset]
    if len(matches) != 1:
        raise SystemExit(f"{subset}: expected one folder, found {matches}")
    return matches[0]


def nested_pool(names, ratios, size, threshold, seed):
    """[(class, tile)] in pick order: round-robin over classes, a random unpicked carrier each."""
    rng = random.Random(seed)
    carriers = {}
    for n in names:
        for cls, r in ratios.get(n, {}).items():
            if cls not in INVALID_CLS and r > threshold:
                carriers.setdefault(cls, []).append(n)
    for c in carriers.values():
        rng.shuffle(c)
    classes = sorted(carriers)
    rng.shuffle(classes)
    picked, order, i = set(), [], 0
    while len(order) < size and any(carriers.values()):
        cls = classes[i % len(classes)]
        i += 1
        while carriers[cls] and carriers[cls][-1] in picked:
            carriers[cls].pop()
        if carriers[cls]:
            tile = carriers[cls].pop()
            picked.add(tile)
            order.append((cls, tile))
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--root-dir", required=True)
    p.add_argument("--pool-subset", required=True, help="subset whose training tiles form the pool")
    p.add_argument("--into-subset", default=None, help="subset whose folder receives the pool; "
                                                       "defaults to --pool-subset")
    p.add_argument("--prefix", required=True, help="pool files are <prefix>_<n>.json")
    p.add_argument("--sizes", default="10,50,100")
    p.add_argument("--source", default="train_samples")
    p.add_argument("--fg-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    src_dir = find_subset(args.root_dir, args.pool_subset)
    dst_dir = find_subset(args.root_dir, args.into_subset or args.pool_subset)
    with open(os.path.join(src_dir, f"{args.pool_subset}.json")) as f:
        ratios = json.load(f)
    with open(os.path.join(src_dir, f"{args.source}.txt")) as f:
        names = sorted(line.strip() for line in f if line.strip())

    sizes = sorted(int(s) for s in args.sizes.split(","))
    order = nested_pool(names, ratios, sizes[-1], args.fg_ratio, args.seed)
    if len(order) < sizes[-1]:
        print(f"  !! only {len(order)} carrier tiles available, pools capped there")
    for n in sizes:
        pool = {}
        for cls, tile in order[:n]:
            pool.setdefault(cls, []).append(tile)
        out = os.path.join(dst_dir, f"{args.prefix}_{n}.json")
        with open(out, "w") as f:
            json.dump(pool, f)
        print(f"  {out}: {min(n, len(order))} tiles, {({k: len(v) for k, v in pool.items()})}")


if __name__ == "__main__":
    main()
