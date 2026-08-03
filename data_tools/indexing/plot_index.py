# -*- coding: utf-8 -*-
"""Which tiles a budget buys, against the whole index: by class, by corpus, or by both.

Every panel is the same picture - what was available in grey, what the budget bought in colour
- asked three ways:

    --group class     one panel per class, each in its own colour. Grey is every tile carrying
                      that class, colour is the ones bought, so each panel is a coverage
                      question about that class alone: of everything that could have been
                      annotated for it, which parts did the budget reach. A tile is
                      multi-label - 3,079 carry forest and 3,028 carry grass, largely the same
                      tiles - so it appears in every panel whose class it carries, which one
                      colour per point could not show.

    --group corpus-class
                      a matrix, a row per corpus and a column per class, coloured by class.
                      This is the grouping `--balance corpus-class` allocates over, so it is
                      the one that shows whether each cell got its own coverage; cells that do
                      not exist, Bayern settlement or Hameln railway, are left blank.

    --group dataset   one panel, one colour per corpus. Every tile belongs to exactly one
                      corpus, so grey is the whole pool and the bought tiles can be coloured
                      together - and the question, does the budget reach Hameln, Bayern
                      (Donauworth) and Siegfried alike or pile into whichever is largest, is
                      read off a single scatter.

The layout defaults to t-SNE, which preserves neighbourhoods so that an empty region in a panel
means the budget really did miss that part of the cell. `--projection pca` preserves distance
instead, but holds a quarter of the variance in two dimensions and hides the gaps it implies.

    PYTHONPATH=. python data_tools/indexing/plot_index.py \\
        -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \\
        --group class --budget 300 -dkw hismap,siegfried -xkw hameln.3921_2023
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_tools.indexing.plot_selection import project
from data_tools.indexing.select_budget import (build_pool, corpus_map, guarded_cells,
                                                load_index, select)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"

# The documented categorical order. Only the first three clear the all-pairs gates a scatter
# needs, which is why the corpora - three of them, one panel - are coloured together while the
# classes get a panel each, so no two class hues ever have to be told apart side by side.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def corpus_of(subset_dir):
    """Which corpus a subset belongs to: hameln / bayern / siegfried.

    The hismap sheets are named `<corpus>.<sheet>_<year>`, so the prefix is the corpus. The
    Siegfried subsets are named for their class instead - `railway`, `vineyard` - and only the
    parent directory says what they are.
    """
    subset = os.path.basename(subset_dir)
    return subset.split(".")[0] if "." in subset else os.path.basename(os.path.dirname(subset_dir))


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


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--group", default="corpus-class",
                        choices=["dataset", "class", "corpus-class"])
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--strategy", default="kcenter",
                        choices=["uniform", "random", "kcenter", "fps", "fps-medoid", "kmeans"])
    parser.add_argument("--balance", default="corpus-class",
                        choices=["none", "class", "corpus", "corpus-class", "subset-class"],
                        help="how the budget being drawn was allocated")
    parser.add_argument("--pool", default="train_samples")
    parser.add_argument("-dkw", "--data-keywords", default="hismap,siegfried")
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
                        help="default results/index_<group>_<strategy>_b<budget>.png")
    args = parser.parse_args()

    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))

    vectors, row_of = load_index(args.index)
    dirs, present_of = build_pool(args.root_dir, row_of, args.pool, fg_ratio,
                                  args.data_keywords, args.exclude_keys)
    present_of = retained_pool(present_of, vectors, row_of, args.balance,
                               corpus_map(dirs), args.density_guard)
    pool_names = sorted(present_of)
    xy, axis_labels = project(vectors[[row_of[n] for n in pool_names]], args.projection,
                              args.seed, args.perplexity, pool_names, args.cache_dir)
    at = {n: i for i, n in enumerate(pool_names)}
    print(f"candidate pool: {len(pool_names)} annotated tiles over {len(dirs)} subsets")

    post_layout = None
    corpus = corpus_map(dirs)
    corpus_of_tile = {n: corpus[n.rsplit(".", 1)[0]] for n in pool_names}

    members = {}
    if args.group == "dataset":
        for n in pool_names:
            members.setdefault(corpus_of_tile[n], []).append(n)
    elif args.group == "class":
        for n, present in present_of.items():
            for cls in present:
                members.setdefault(cls, []).append(n)
    else:                                        # corpus-class: one panel per (corpus, class)
        for n, present in present_of.items():
            for cls in present:
                members.setdefault((corpus_of_tile[n], cls), []).append(n)

    bought = select(present_of, vectors, row_of, args.budget, args.strategy,
                    balance=args.balance, corpus_of=corpus,
                    density_guard=args.density_guard, seed=args.seed, verbose=False)
    # a pair is bought for one class, so a class-bearing panel must match on the pair, not the
    # tile - a tile bought for water is not annotated for grass and must not be drawn as such
    bought_by_class, bought_tiles = {}, {t for t, _ in bought}
    for tile, cls in bought:
        bought_by_class.setdefault(cls, set()).add(tile)
    variant = "" if args.balance == "none" else f", balanced by {args.balance}"
    print(f"{args.strategy}{variant} at budget {args.budget}: "
          f"{len(bought)} pairs over {len(bought_tiles)} tiles")

    def is_bought(g, tile):
        cls = g[1] if isinstance(g, tuple) else (g if args.group == "class" else None)
        return tile in (bought_by_class.get(cls, ()) if cls else bought_tiles)

    # Biggest group first, so the rare ones are drawn on top rather than buried.
    groups = sorted(members, key=lambda g: -len(members[g]))
    picked = {g: [n for n in members[g] if is_bought(g, n)] for g in groups}
    for g in groups:
        label = "/".join(g) if isinstance(g, tuple) else g
        print(f"    {label:<22} {len(picked[g]):>4} bought of {len(members[g]):>6} candidate pairs")
    if args.group == "dataset" and len(groups) > 3:
        print(f"!! {len(groups)} corpora share one panel; only three hues validate for a "
              f"scatter, so read this one with the legend")

    def draw(ax, g, colour):
        """This group's own candidates in grey, and the ones the budget bought over them.

        The backdrop is the group's candidates rather than the whole pool, so each panel is
        read as a coverage question about that group alone: of everything that could have been
        annotated for it, which parts did the budget reach.
        """
        avail = np.array([at[n] for n in members[g]])
        ax.scatter(xy[avail, 0], xy[avail, 1], s=4, c=MUTED, alpha=0.22,
                   linewidths=0, rasterized=True)
        if not picked[g]:
            return
        idx = np.array([at[n] for n in picked[g]])
        # a surface-coloured ring, so bought tiles stay countable where they pile up
        ax.scatter(xy[idx, 0], xy[idx, 1], s=20, c=colour,
                   edgecolors=SURFACE, linewidths=0.5, zorder=3)

    if args.group == "dataset":
        fig, ax = plt.subplots(figsize=(8.4, 6.6 + 1.6))
        fig.patch.set_facecolor(SURFACE)
        style_axes(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=MUTED, alpha=0.16,
                   linewidths=0, rasterized=True)
        for i, g in enumerate(groups):
            if picked[g]:
                idx = np.array([at[n] for n in picked[g]])
                ax.scatter(xy[idx, 0], xy[idx, 1], s=22, c=SLOTS[i % len(SLOTS)],
                           edgecolors=SURFACE, linewidths=0.5, zorder=3)
        ax.set_xlabel(axis_labels[0], fontsize=9, color=MUTED)
        ax.set_ylabel(axis_labels[1], fontsize=9, color=MUTED)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                              markerfacecolor=MUTED, markeredgecolor="none", alpha=0.6,
                              label=f"candidate pool ({len(pool_names)})")]
        handles += [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                               markerfacecolor=SLOTS[i % len(SLOTS)], markeredgecolor=SURFACE,
                               label=f"{g}  {len(picked[g])}/{len(members[g])}")
                    for i, g in enumerate(groups)]
        title = f"Which tiles a budget of {args.budget} buys, by corpus"
    elif args.group == "corpus-class":
        # A matrix: a row per corpus, a column per class, coloured by class so a column reads
        # as one thing. Cells that do not exist - Bayern has no settlement, Hameln no railway -
        # are left blank rather than drawn empty, which would read as "bought nothing".
        corpora = sorted({c for c, _ in members}, key=lambda c: -sum(
            len(v) for k, v in members.items() if k[0] == c))
        classes = sorted({c for _, c in members}, key=lambda c: -sum(
            len(v) for k, v in members.items() if k[1] == c))
        fig, axes = plt.subplots(len(corpora), len(classes),
                                 figsize=(2.5 * len(classes) + 0.8, 2.5 * len(corpora) + 1.9),
                                 sharex=True, sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for r, corp in enumerate(corpora):
            for c, cls in enumerate(classes):
                ax = axes[r][c]
                if (corp, cls) not in members:
                    ax.set_visible(False)
                    continue
                style_axes(ax)
                draw(ax, (corp, cls), SLOTS[c % len(SLOTS)])
                ax.text(0.03, 0.97, f"{len(picked[(corp, cls)])}/{len(members[(corp, cls)])}",
                        transform=ax.transAxes, fontsize=9.5, color=INK_2, va="top")
        def label_matrix():
            """Headers anchored to the grid, not to a cell.

            A cell that does not exist is hidden, and half the headers would go with it -
            there is no hameln/railway to hang the `railway` column on. Hidden axes keep their
            position, so the labels are placed off the grid geometry after the layout settles.
            """
            for c, cls in enumerate(classes):
                pos = axes[0][c].get_position()
                fig.add_artist(plt.Line2D([pos.x0 + 0.004], [pos.y1 + 0.026], marker="o",
                                          markersize=6, linestyle="",
                                          markerfacecolor=SLOTS[c % len(SLOTS)],
                                          markeredgecolor="none", transform=fig.transFigure))
                fig.text(pos.x0 + 0.014, pos.y1 + 0.018, cls, fontsize=11,
                         color=INK, fontweight="bold", va="bottom")
            for r, corp in enumerate(corpora):
                pos = axes[r][0].get_position()
                fig.text(0.012, (pos.y0 + pos.y1) / 2, corp, fontsize=11, color=INK,
                         fontweight="bold", rotation=90, va="center", ha="left")

        post_layout = label_matrix
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                              markerfacecolor=MUTED, markeredgecolor="none", alpha=0.6,
                              label="every tile of that corpus carrying that class"),
                   plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                              markerfacecolor=INK_2, markeredgecolor=SURFACE,
                              label="the ones the budget bought")]
        title = f"Which tiles a budget of {args.budget} buys, per corpus and class"
    else:
        cols = min(len(groups), 3)
        rows = (len(groups) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows + 1.6),
                                 sharex=True, sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, (i, g) in zip(axes.ravel(), enumerate(groups)):
            style_axes(ax)
            draw(ax, g, SLOTS[i % len(SLOTS)])
            # a colour chip beside the label, so the title itself stays in ink
            ax.scatter([0.012], [1.055], s=42, c=SLOTS[i % len(SLOTS)], linewidths=0,
                       transform=ax.transAxes, clip_on=False)
            ax.text(0.05, 1.03, g, transform=ax.transAxes, fontsize=12,
                    color=INK, fontweight="bold", va="bottom")
            ax.text(0.05, 1.005, f"{len(picked[g])} bought of {len(members[g])} candidates",
                    transform=ax.transAxes, fontsize=9, color=INK_2, va="top")
        for ax in axes.ravel()[len(groups):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(axis_labels[0], fontsize=9, color=MUTED)
        for row in axes:
            row[0].set_ylabel(axis_labels[1], fontsize=9, color=MUTED)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=5,
                              markerfacecolor=MUTED, markeredgecolor="none", alpha=0.6,
                              label="every tile carrying the class"),
                   plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                              markerfacecolor=INK_2, markeredgecolor=SURFACE,
                              label="the ones the budget bought, in the class's colour")]
        title = f"Which tiles a budget of {args.budget} buys, by class"

    legend = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.01),
                        ncol=len(handles), frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_2)

    # Reserved in inches, not figure fractions, so the title block holds at any panel count.
    fig_h = fig.get_figheight()
    fig.suptitle(title, fontsize=14, color=INK, x=0.02, ha="left", y=1 - 0.34 / fig_h)
    fig.text(0.02, 1 - 0.62 / fig_h, f"{args.strategy}{variant}, DINOv2 CLS space, {len(pool_names)} "
                          f"candidate tiles over {len(dirs)} subsets, laid out by "
                          f"{args.projection}",
             fontsize=9.5, color=INK_2, ha="left")

    out = args.output or (f"results/index_{args.group}_{args.strategy}"
                          f"_{args.balance}_b{args.budget}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    head = 1.25 if args.group == "corpus-class" else 0.95     # matrix adds column headers
    fig.tight_layout(rect=(0.03 if args.group == "corpus-class" else 0, 0.62 / fig_h,
                           1, 1 - head / fig_h))
    if post_layout:
        post_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
