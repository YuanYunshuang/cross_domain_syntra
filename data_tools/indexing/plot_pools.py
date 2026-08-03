# -*- coding: utf-8 -*-
"""Show what a retrieval pool actually contains, one row per pool.

`--cell-gap` unions sheets whose mean DINOv2 embeddings lie within that cosine distance, and the
grouping is only as good as the tiles it puts together. This draws them: one row per pool, one
tile per member sheet, so a pool that mixes drawing conventions is visible rather than inferred
from a distance matrix.

Read it against the gap sweep. At 0 every sheet is its own pool and the question does not arise;
as the gap grows the rows should stay stylistically of a piece, and the gap to use is the largest
one where they still do.

    PYTHONPATH=. python data_tools/indexing/plot_pools.py \\
        -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \\
        --cell-gap 0.05 -dkw hismap,siegfried -xkw hameln.3921_2023
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_tools.indexing.select_budget import build_pool, corpus_map, load_index, style_map

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"


def tile_path(subset_dir, tile):
    """`hameln.3821_1974.0_1920` lives at `<subset>/imgs/0_1920.png`."""
    return os.path.join(subset_dir, "imgs", tile.rsplit(".", 1)[1] + ".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--cell-gap", type=float, default=0.05)
    parser.add_argument("--merge-corpora", action="store_true")
    parser.add_argument("--per-pool", type=int, default=8, help="tiles shown per pool")
    parser.add_argument("--pool", default="train_samples")
    parser.add_argument("-dkw", "--data-keywords", default="hismap,siegfried")
    parser.add_argument("-xkw", "--exclude-keys", default="hameln.3921_2023")
    parser.add_argument("--fg-ratio", default="0.05")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))
    vectors, row_of = load_index(args.index)
    dirs, present_of = build_pool(args.root_dir, row_of, args.pool, fg_ratio,
                                  args.data_keywords, args.exclude_keys)
    dir_of = {os.path.basename(d): d for d in dirs}
    by_sheet = {}
    for name in present_of:
        by_sheet.setdefault(name.rsplit(".", 1)[0], []).append(name)

    cells = style_map(by_sheet, vectors, row_of, corpus_map(dirs),
                      args.cell_gap, args.merge_corpora)
    pools = {}
    for sheet, tag in cells.items():
        pools.setdefault(tag, []).append(sheet)
    order = sorted(pools, key=lambda t: (-len(pools[t]), t))
    print(f"cell gap {args.cell_gap:g}: {len(pools)} pools over {len(cells)} sheets")

    rng = np.random.RandomState(args.seed)
    rows = []
    for tag in order:
        members = sorted(pools[tag])
        # one tile per member sheet first, so a mixed pool shows its mixture rather than
        # whichever sheet happens to be largest
        picks, i = [], 0
        while len(picks) < args.per_pool:
            sheet = members[i % len(members)]
            i += 1
            pool_tiles = [t for t in by_sheet[sheet] if t not in picks]
            if pool_tiles:
                picks.append(pool_tiles[int(rng.randint(len(pool_tiles)))])
            if i > len(members) * args.per_pool:
                break
        rows.append((tag, members, picks))

    fig, axes = plt.subplots(len(rows), args.per_pool,
                             figsize=(1.55 * args.per_pool + 3.0, 1.62 * len(rows) + 1.1),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (tag, members, picks) in enumerate(rows):
        for c in range(args.per_pool):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(picks):
                sheet = picks[c].rsplit(".", 1)[0]
                path = tile_path(dir_of[sheet], picks[c])
                if os.path.isfile(path):
                    ax.imshow(Image.open(path).convert("RGB").resize((256, 256)))
                ax.set_xlabel(sheet.split(".")[-1], fontsize=6.5, color=MUTED, labelpad=2)
        label = f"{len(members)} sheet" + ("s" if len(members) > 1 else "")
        axes[r][0].set_ylabel(f"{label}\n" + "\n".join(m.split(".")[-1] for m in members[:5])
                              + ("\n…" if len(members) > 5 else ""),
                              fontsize=7, color=INK, rotation=0, ha="right", va="center",
                              labelpad=8)

    fig_h = fig.get_figheight()
    fig.suptitle(f"Retrieval pools at cell-gap {args.cell_gap:g} — {len(pools)} pools",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1 - 0.30 / fig_h)
    fig.text(0.01, 1 - 0.56 / fig_h,
             "one row per pool, one tile per member sheet; a row that mixes drawing "
             "conventions is a pool that should not have merged",
             fontsize=9, color=INK_2, ha="left")
    out = args.output or f"results/pools_gap{args.cell_gap:g}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.80 / fig_h))
    fig.savefig(out, dpi=125, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
