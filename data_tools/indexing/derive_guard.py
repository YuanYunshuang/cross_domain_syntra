# -*- coding: utf-8 -*-
"""Re-derive the density-guard level in tile units (PAPER_PLAN.md section 6b).

The guard was chosen against a *pass*-denominated budget: random selection under corpus-class
balancing puts 18% +- 2 of its picks in the globally-sparsest tenth of the candidate pairs, and
g=15 was the smallest guard bringing farthest-point sampling back to that reference at B=300 and
B=1000 pairs. Section 3 changes the budget unit to one tile labelled for every class in its
sheet's vocabulary, so both sides of that comparison move: the pool is counted in tiles, a tile
is charged once however many cells want it, and at a given B there are more classes per unit.
The guard level is a property of the budget, not of the method, so it has to be re-derived rather
than carried over - and deciding it after seeing the budget curve would be choosing on the
outcome.

The selection policy is untouched: this calls `select()` with the same balance, cell gap, seed
and floor, and changes only the meter. The pair budget is searched for the value that spends
exactly B distinct tiles, so "B tiles" means what it says while every allocation decision inside
stays the one that was validated.

Run: PYTHONPATH=. python data_tools/indexing/derive_guard.py
"""
import argparse
import json

import numpy as np

from data_tools.indexing.select_budget import (build_pool, cells_of, guarded_cells,
                                               load_index, local_density, select)


def tiles_of(pairs):
    return {t for t, _ in pairs}


def spend_tiles(present_of, vectors, row_of, budget_tiles, strategy, guard, args):
    """Smallest pair budget whose selection holds `budget_tiles` distinct tiles.

    Distinct tiles rise monotonically with the pair budget under a fixed seed, so a bisection is
    exact rather than approximate. Returns (pairs, tiles, selection).
    """
    lo, hi = budget_tiles, max(budget_tiles * 6, 64)
    best = None
    for _ in range(24):
        sel = select(present_of, vectors, row_of, hi, strategy, gap=args.cell_gap,
                     balance=args.balance, corpus_of=None, density_guard=guard,
                     min_per_group=args.min_per_group, seed=args.seed, verbose=False)
        if len(tiles_of(sel)) >= budget_tiles:
            break
        lo, hi = hi, hi * 2
    while lo <= hi:
        mid = (lo + hi) // 2
        sel = select(present_of, vectors, row_of, mid, strategy, gap=args.cell_gap,
                     balance=args.balance, corpus_of=None, density_guard=guard,
                     min_per_group=args.min_per_group, seed=args.seed, verbose=False)
        n = len(tiles_of(sel))
        if n >= budget_tiles:
            best = (mid, sel)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        sel = select(present_of, vectors, row_of, 10 ** 6, strategy, gap=args.cell_gap,
                     balance=args.balance, corpus_of=None, density_guard=guard,
                     min_per_group=args.min_per_group, seed=args.seed, verbose=False)
        best = (len(sel), sel)
    return best[0], tiles_of(best[1]), best[1]


def covering_radii(selection, cells, vectors, row_of):
    """Per cell, the cosine distance from its worst-served candidate to its nearest pick."""
    chosen = tiles_of(selection)
    worst, means = [], []
    for pairs in cells.values():
        pool = sorted({t for t, _ in pairs})
        picks = sorted(chosen & set(pool))
        if not picks or len(pool) < 2:
            continue
        d = 1.0 - vectors[[row_of[t] for t in pool]] @ vectors[[row_of[t] for t in picks]].T
        near = d.min(axis=1)
        worst.append(float(near.max()))
        means.append(float(near.mean()))
    return (max(worst) if worst else float("nan"),
            float(np.mean(means)) if means else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--root-dir", default="/home/yuan/data/syntra")
    p.add_argument("--index", default="/home/yuan/data/syntra/global_index.npz")
    p.add_argument("-dkw", "--data-keywords", default="hismap,siegfried")
    p.add_argument("-xkw", "--exclude-keys", default=None,
                   help="hameln.3921_2023 is KEPT by decision (PAPER_PLAN section 9)")
    p.add_argument("--pool", default="train_samples")
    p.add_argument("--fg-ratio", type=float, default=0.05)
    p.add_argument("--cell-gap", type=float, default=0.0)
    p.add_argument("--balance", default="cell-class")
    p.add_argument("--min-per-group", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--budgets", default="100,300,1000")
    p.add_argument("--guards", default="0,3,5,10,15,20")
    p.add_argument("--neutral-seeds", type=int, default=5)
    p.add_argument("--decile", type=float, default=10.0)
    p.add_argument("--out", default="results/guard_tile_units.json")
    args = p.parse_args()

    vectors, row_of = load_index(args.index)
    _, present_of = build_pool(args.root_dir, row_of, args.pool, args.fg_ratio,
                               args.data_keywords, args.exclude_keys)
    pool_tiles = sorted(present_of)
    print(f"candidate pool: {len(pool_tiles)} tiles, "
          f"{sum(len(v) for v in present_of.values())} (tile, class) pairs")

    # "Sparsest tenth" is global and over tiles, the population the budget now counts.
    dens = local_density(vectors[[row_of[t] for t in pool_tiles]], 20)
    cut = np.percentile(dens, 100 - args.decile)
    sparse = {t for t, r in zip(pool_tiles, dens) if r >= cut}
    print(f"globally sparsest {args.decile:g}% of tiles: {len(sparse)}")

    cells = cells_of(present_of, args.cell_gap, args.balance, None, vectors, row_of)
    print(f"{len(cells)} cells under balance={args.balance}, gap={args.cell_gap:g}\n")

    rows, report = [], {}
    for B in [int(x) for x in args.budgets.split(",")]:
        # neutral reference: random, unguarded, averaged over seeds
        neutral = []
        for s in range(args.neutral_seeds):
            a = argparse.Namespace(**{**vars(args), "seed": args.seed + s})
            _, tiles, _ = spend_tiles(present_of, vectors, row_of, B, "random", 0, a)
            neutral.append(100.0 * len(tiles & sparse) / max(len(tiles), 1))
        ref, ref_sd = float(np.mean(neutral)), float(np.std(neutral))
        print(f"B={B} tiles | density-neutral reference: {ref:.1f}% +- {ref_sd:.1f} "
              f"in the sparsest {args.decile:g}%")
        report[B] = {"reference": ref, "reference_sd": ref_sd, "guards": {}}
        print(f"  {'guard':>5} {'pairs':>7} {'tiles':>6} {'sparsest':>9} {'vs neutral':>11} "
              f"{'worst cell':>11} {'mean cell':>10}")
        for g in [float(x) for x in args.guards.split(",")]:
            pairs, tiles, sel = spend_tiles(present_of, vectors, row_of, B, "fps", g, args)
            share = 100.0 * len(tiles & sparse) / max(len(tiles), 1)
            gc = guarded_cells(present_of, vectors, row_of, args.cell_gap, args.balance,
                               None, g, verbose=False)
            worst, mean = covering_radii(sel, gc, vectors, row_of)
            ratio = share / ref if ref else float("nan")
            print(f"  {g:>5g} {pairs:>7} {len(tiles):>6} {share:>8.1f}% {ratio:>10.2f}x "
                  f"{worst:>11.3f} {mean:>10.3f}")
            report[B]["guards"][g] = {"pairs": pairs, "tiles": len(tiles), "sparsest": share,
                                      "vs_neutral": ratio, "worst_cell": worst,
                                      "mean_cell": mean}
            rows.append((B, g, ratio))
        print()

    print("smallest guard reaching the neutral reference (<= 1.1x), per rung:")
    for B in report:
        ok = [g for g, v in sorted(report[B]["guards"].items()) if v["vs_neutral"] <= 1.1]
        print(f"  B={B:>5} tiles: g={ok[0]:g}" if ok else f"  B={B:>5} tiles: none in range")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
