# -*- coding: utf-8 -*-
"""Where each strategy spends its budget, drawn in DINOv2 space (E13).

One panel per strategy: the whole candidate pool in grey, the tiles the strategy bought in
blue. This is the picture behind the risk the plan names - k-center maximises spread and so
gravitates to outliers, while k-means centroids trade coverage for typicality - and it is read
off the shape of the blue set rather than argued.

The layout defaults to t-SNE. PCA (`--projection pca`) preserves distance but its two leading
components hold only about a quarter of the variance here, which makes the picture lie by
omission: regions that look unvisited are covered by tiles the projection puts elsewhere. t-SNE
preserves neighbourhoods instead, so an empty region means something - at the cost of axes with
no scale. Either way the counts and the covering radius come from the 384-d vectors, not the
picture.

Each panel carries the coverage radius: the cosine distance from a pool tile to its nearest
bought tile, as a mean and as a worst case. The mean is how well the corpus is served on
average; the worst case is the covering radius, the quantity k-center greedy minimises. The
rarest class's share is printed next to them, because a coverage-driven selection is
class-blind and can starve a minority class - which is what `--balance` exists to fix.

The selection is imported from `select_budget.py`, so the tiles drawn here are the tiles that
would be written by the same arguments - not a second implementation of the same idea.

    PYTHONPATH=. python data_tools/indexing/plot_selection.py \\
        -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \\
        --budget 300 -dkw hismap -xkw hameln.3921_2023 -o results/selection_b300.png
"""

import argparse
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_tools.indexing.select_budget import (build_pool, corpus_map, guarded_cells,
                                                load_index, select)

# Chart chrome and the one series colour. The pool is context, not a second series: it wears
# the muted axis token, so only the bought tiles read as data.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"

STRATEGIES = ["uniform", "random", "fps", "fps-medoid"]


def pca(vectors, dims=2):
    """Project onto the leading principal components; returns (coords, explained fractions).

    Plain SVD rather than sklearn, so this works in any environment and is deterministic - the
    figure must not move between runs.
    """
    centred = vectors - vectors.mean(axis=0)
    _, sv, components = np.linalg.svd(centred, full_matrices=False)
    variance = sv ** 2
    return centred @ components[:dims].T, variance[:dims] / variance.sum()


def pca2(vectors):
    """Back-compatible two-component PCA."""
    return pca(vectors, 2)


def project(vectors, method="tsne", seed=2026, perplexity=30, names=None, cache_dir=None):
    """Lay the pool out in two dimensions; returns (xy, axis labels).

    PCA is honest about distance but hopeless at showing neighbourhoods here - its two leading
    components hold about a quarter of the variance, so regions that look unvisited generally
    are not. t-SNE keeps neighbours together, which is what makes a gap in the picture mean
    something, at the price of axes that carry no scale and distances that cannot be compared
    across the plot. Read counts and covering radii from the numbers either way.

    t-SNE is preceded by a PCA to 50 dimensions, the standard recipe, and cached: it takes
    minutes over 10k tiles and every figure would otherwise pay for it again.
    """
    if method == "pca":
        xy, explained = pca(vectors, 2)
        return xy, (f"PC1  ({100 * explained[0]:.0f}% var)",
                    f"PC2  ({100 * explained[1]:.0f}% var)")

    key = None
    if cache_dir and names is not None:
        digest = hashlib.md5(("\n".join(names)).encode()).hexdigest()[:12]
        key = os.path.join(cache_dir, f"tsne_{digest}_p{perplexity}_s{seed}.npy")
        if os.path.isfile(key):
            return np.load(key), ("t-SNE 1", "t-SNE 2")

    from sklearn.manifold import TSNE
    pre, _ = pca(vectors, min(50, vectors.shape[1]))
    print(f"running t-SNE over {len(vectors)} tiles (perplexity {perplexity}) ...")
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca", metric="cosine",
              random_state=seed).fit_transform(pre.astype(np.float32))
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(key, xy)
    return xy, ("t-SNE 1", "t-SNE 2")


def retained_pool(present_of, vectors, row_of, balance, corpus_of, density_guard, verbose=True):
    """The candidacy the selector actually saw, as {tile: [classes still in play]}.

    The guard is per cell and the unit is a (tile, class) pair, so a tile can stay a candidate
    for the class it is typical of and be dropped for one it is not. The figures have to show
    that same candidacy or the grey backdrop would advertise pairs the budget could not buy.
    """
    cells = guarded_cells(present_of, vectors, row_of, balance, corpus_of,
                          density_guard, verbose=verbose)
    kept = {}
    for pairs in cells.values():
        for tile, cls in pairs:
            kept.setdefault(tile, set()).add(cls)
    return {t: sorted(cs) for t, cs in kept.items()}


def coverage(pool_vectors, chosen_vectors):
    """Cosine distance from every pool tile to its nearest bought tile: (mean, worst).

    The worst case is the covering radius, which is the quantity k-center greedy actually
    minimises, and the two numbers disagree by design: k-center pays a worse mean to leave no
    corner of the corpus unserved, while k-means buys typical tiles and abandons the fringe.
    A percentile would hide exactly that, since the fringe is where it lives.
    """
    nearest = (1.0 - pool_vectors @ chosen_vectors.T).min(axis=1)
    return float(nearest.mean()), float(nearest.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--strategies", default=",".join(STRATEGIES),
                        help="comma-separated subset of " + "/".join(STRATEGIES))
    parser.add_argument("--balance", default="none",
                        choices=["none", "class", "corpus", "corpus-class", "subset-class"],
                        help="draw the variant that splits the budget over these cells")
    parser.add_argument("--pool", default="train_samples")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--fg-ratio", default="0.05")
    parser.add_argument("--density-guard", type=float, default=0,
                        help="drop the sparsest N%% of each cell from candidacy first; 10 "
                             "stops farthest-point sampling spending the budget on outliers")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--projection", default="tsne", choices=["tsne", "pca"],
                        help="t-SNE keeps neighbourhoods, so a gap in the picture means "
                             "something; PCA keeps distances but hides them")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--cache-dir", default="results/.projections")
    parser.add_argument("-o", "--output", default=None,
                        help="default results/selection_b<budget>[_<balance>].png")
    args = parser.parse_args()

    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    vectors, row_of = load_index(args.index)
    dirs, present_of = build_pool(args.root_dir, row_of, args.pool, fg_ratio,
                                  args.data_keywords, args.exclude_keys)
    present_of = retained_pool(present_of, vectors, row_of, args.balance,
                               corpus_map(dirs), args.density_guard)
    pool_names = sorted(present_of)
    pool_vectors = vectors[[row_of[n] for n in pool_names]]
    xy, axis_labels = project(pool_vectors, args.projection, args.seed,
                              args.perplexity, pool_names, args.cache_dir)
    at = {n: i for i, n in enumerate(pool_names)}
    print(f"candidate pool: {len(pool_names)} annotated tiles over {len(dirs)} subsets")

    # The class most at risk of being skipped is the one with fewest candidates.
    counts = {}
    for present in present_of.values():
        for cls in present:
            counts[cls] = counts.get(cls, 0) + 1
    rarest = min(counts, key=counts.get)

    picked = {}
    for strategy in strategies:
        chosen = select(present_of, vectors, row_of, args.budget, strategy,
                        balance=args.balance, corpus_of=corpus_map(dirs),
                        density_guard=args.density_guard, seed=args.seed, verbose=False)
        tiles = sorted({t for t, _ in chosen})
        mean_d, worst_d = coverage(pool_vectors, vectors[[row_of[t] for t in tiles]])
        share = sum(1 for _, c in chosen if c == rarest)
        picked[strategy] = (chosen, tiles, mean_d, worst_d, share)
        print(f"    {strategy:<11} {len(chosen):>5} pairs over {len(tiles):>4} tiles   "
              f"coverage mean {mean_d:.3f} worst {worst_d:.3f}   {rarest} {share}")

    cols = min(len(strategies), 2)
    rows = (len(strategies) + cols - 1) // cols
    # the title block and legend get their own inches on top of the panels, not a slice of them
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.7 * rows + 1.6),
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, strategy in zip(axes.ravel(), strategies):
        chosen, tiles, mean_d, worst_d, share = picked[strategy]
        sel = np.array([at[t] for t in tiles])
        ax.set_facecolor(SURFACE)
        # equal aspect: the figure is read as distances, and the covering radius printed above
        # would be a lie on axes at different scales
        ax.set_aspect("equal")
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=MUTED, alpha=0.25,
                   linewidths=0, rasterized=True)
        # a surface-coloured ring, so bought tiles stay countable where they pile up
        ax.scatter(xy[sel, 0], xy[sel, 1], s=20, c=SERIES,
                   edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.set_title(strategy, fontsize=12, color=INK, fontweight="bold",
                     loc="left", pad=14)
        ax.text(0, 1.015, f"{len(chosen)} pairs / {len(tiles)} tiles   "
                          f"worst {worst_d:.3f}   {rarest} {share}",
                transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS)
            spine.set_linewidth(0.8)

    for ax in axes.ravel()[len(strategies):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(axis_labels[0], fontsize=9, color=MUTED)
    for row in axes:
        row[0].set_ylabel(axis_labels[1], fontsize=9, color=MUTED)

    variant = "" if args.balance == "none" else f" balanced by {args.balance}"
    # The title block and legend are reserved in inches, not in figure fractions: a one-row
    # figure is half the height of a two-row one and fixed fractions collide with the axes.
    fig_h = fig.get_figheight()
    fig.suptitle(f"Where a budget of {args.budget} tiles is spent{variant}",
                 fontsize=14, color=INK, x=0.02, ha="left", y=1 - 0.34 / fig_h)
    fig.text(0.02, 1 - 0.62 / fig_h, f"DINOv2 CLS space, {len(pool_names)} candidate tiles over "
                          f"{len(dirs)} subsets, laid out by {args.projection}",
             fontsize=9.5, color=INK_2, ha="left")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                          markerfacecolor=MUTED, markeredgecolor="none", alpha=0.6,
                          label=f"candidate pool ({len(pool_names)})"),
               plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=SERIES, markeredgecolor=SURFACE,
                          label=f"selected ({args.budget})")]
    legend = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.01),
                        ncol=2, frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_2)

    out = args.output or (f"results/selection_b{args.budget}"
                          f"{'' if args.balance == 'none' else '_' + args.balance}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0.62 / fig_h, 1, 1 - 0.95 / fig_h))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
