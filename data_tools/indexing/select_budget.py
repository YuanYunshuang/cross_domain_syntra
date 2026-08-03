# -*- coding: utf-8 -*-
"""Spend a fixed annotation budget: which tiles are worth labelling (E11).

Given a total budget of B tiles, choose which ones to annotate. The selection is written in the
`cls_selection_*.json` format, so it can be used either as a training set or as a retrieval
pool by passing its name to `make_pool_pairing.py --source`.

Strategies:

    uniform   n tiles per class and subset - the n-shot scheme used so far, which spends the
              budget evenly regardless of how redundant a sheet is
    random    B tiles drawn uniformly from the candidate pool
    kcenter   farthest-point (k-center greedy) selection in DINOv2 space: repeatedly take the
              tile furthest from everything already chosen, maximising coverage of the corpus
    kmeans    k-means centroids, each replaced by its nearest real tile; trades coverage for
              typicality, and is the guard against k-center chasing outliers

Candidates are restricted to tiles that actually contain an annotated class above the
foreground threshold. Without that, coverage-seeking strategies drift to sheet borders, blank
tiles and scanning artefacts - the emptiest tiles are exactly the ones furthest from everything
else.

    PYTHONPATH=. python data_tools/indexing/select_budget.py \\
        -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \\
        --budget 300 --strategy kcenter -dkw hismap,siegfried -xkw hameln.3921_2023
"""

import argparse
import json
import os

import numpy as np

from data_tools.indexing.build_global_index import subset_dirs
from data_tools.indexing.make_pool_pairing import fg_threshold, load_samples

INVALID_CLS = ("background", "nonlabeled")


def kcenter_greedy(vectors, budget, seed):
    """Farthest-point sampling: each step adds the point furthest from the chosen set."""
    rng = np.random.RandomState(seed)
    n = len(vectors)
    first = int(rng.randint(n))
    chosen = [first]
    # cosine distance to the nearest chosen point, updated incrementally
    nearest = 1.0 - vectors @ vectors[first]
    while len(chosen) < min(budget, n):
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, 1.0 - vectors @ vectors[pick])
    return chosen


def kmeans_medoids(vectors, budget, seed, iters=25):
    """k-means on the unit sphere; every centroid is replaced by its nearest real tile."""
    rng = np.random.RandomState(seed)
    n = len(vectors)
    k = min(budget, n)
    # k-means++ style seeding keeps the initial centroids apart
    centroids = [vectors[int(rng.randint(n))]]
    for _ in range(k - 1):
        d = np.min(np.stack([1.0 - vectors @ c for c in centroids]), axis=0)
        probs = np.clip(d, 0, None)
        total = probs.sum()
        idx = int(rng.choice(n, p=probs / total)) if total > 0 else int(rng.randint(n))
        centroids.append(vectors[idx])
    centroids = np.stack(centroids)

    for _ in range(iters):
        assign = np.argmax(vectors @ centroids.T, axis=1)
        moved = False
        for j in range(k):
            members = vectors[assign == j]
            if len(members) == 0:
                continue
            new = members.mean(axis=0)
            new /= np.linalg.norm(new) + 1e-12
            if not np.allclose(new, centroids[j]):
                centroids[j] = new
                moved = True
        if not moved:
            break

    medoids, taken = [], set()
    for j in range(k):
        order = np.argsort(-(vectors @ centroids[j]))
        for idx in order:
            if int(idx) not in taken:
                taken.add(int(idx))
                medoids.append(int(idx))
                break
    return medoids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--budget", type=int, required=True,
                        help="total tiles to annotate (per class and subset for 'uniform')")
    parser.add_argument("--strategy", default="kcenter",
                        choices=["uniform", "random", "kcenter", "kmeans"])
    parser.add_argument("--pool", default="train_samples",
                        help="sample list the budget may be spent from")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--fg-ratio", default="0.05",
                        help="float, or key=value pairs, e.g. hameln=0.05,siegfried=0.01")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default=None, help="output name; default budget_<strategy>_<B>")
    parser.add_argument("--output-root", default=None,
                        help="write into a parallel tree instead of the dataset itself")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))

    store = np.load(args.index, allow_pickle=True)
    names = list(store["names"])
    vectors = store["vectors"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    row_of = {n: i for i, n in enumerate(names)}

    dirs = subset_dirs(args.root_dir, args.data_keywords, args.exclude_keys)
    ratios, classes_of, pool = {}, {}, []
    for subset_dir in dirs:
        subset = os.path.basename(subset_dir)
        with open(os.path.join(subset_dir, f"{subset}.json")) as f:
            ratios.update(json.load(f))
        with open(os.path.join(subset_dir, "color_map.json")) as f:
            classes_of[subset] = {c for c in json.load(f) if c not in INVALID_CLS}
        listing = os.path.join(subset_dir, args.pool)
        listing += ".json" if os.path.isfile(listing + ".json") else ".txt"
        if not os.path.isfile(listing):
            continue
        threshold = fg_threshold(subset, fg_ratio)
        for name in load_samples(listing):
            if name not in row_of:
                continue
            present = [c for c, r in ratios.get(name, {}).items()
                       if c in classes_of[subset] and r > threshold]
            if present:                      # never spend budget on an empty tile
                pool.append((name, present))

    print(f"candidate pool: {len(pool)} annotated tiles over {len(dirs)} subsets")
    rng = np.random.RandomState(args.seed)

    if args.strategy == "uniform":
        by_group = {}
        for name, present in pool:
            subset = name.rsplit(".", 1)[0]
            for cls in present:
                by_group.setdefault((subset, cls), []).append(name)
        picked = set()
        for group, members in sorted(by_group.items()):
            take = min(args.budget, len(members))
            picked.update(rng.choice(members, take, replace=False).tolist())
        selected = sorted(picked)
    else:
        pool_names = [n for n, _ in pool]
        sub = vectors[[row_of[n] for n in pool_names]]
        if args.strategy == "random":
            idx = rng.choice(len(pool_names), min(args.budget, len(pool_names)), replace=False)
        elif args.strategy == "kcenter":
            idx = kcenter_greedy(sub, args.budget, args.seed)
        else:
            idx = kmeans_medoids(sub, args.budget, args.seed)
        selected = sorted(pool_names[int(i)] for i in idx)

    present_of = dict(pool)
    by_class = {}
    for name in selected:
        for cls in present_of[name]:
            by_class.setdefault(cls, []).append(name)

    subsets_hit = {n.rsplit(".", 1)[0] for n in selected}
    print(f"strategy={args.strategy} budget={args.budget}: {len(selected)} tiles, "
          f"{len(subsets_hit)}/{len(dirs)} subsets covered")
    for cls, members in sorted(by_class.items()):
        print(f"    {cls:<14} {len(members):>5} tiles")

    name = args.name or f"budget_{args.strategy}_{args.budget}"
    if args.dry_run:
        print("\n(dry run, nothing written)")
        return

    for subset_dir in dirs:
        subset = os.path.basename(subset_dir)
        out_dir = subset_dir
        if args.output_root:
            from data_tools.indexing.make_pool_pairing import mirror_subset
            out_dir = mirror_subset(subset_dir, args.output_root, subset, args.root_dir)
        # Each subset gets only its own selected tiles. make_pool_pairing keys candidates by
        # the subset that owns the pool file, so a shared file would offer foreign tiles as
        # sources at subset scope and the loader would not have those subsets open at all.
        own = {c: [n for n in v if n.rsplit(".", 1)[0] == subset] for c, v in by_class.items()}
        own = {c: v for c, v in own.items() if v}
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(own, f, indent=2)
    print(f"\nWrote {name}.json to {len(dirs)} subsets")


if __name__ == "__main__":
    main()
