# -*- coding: utf-8 -*-
"""Spend a fixed annotation budget: which tiles are worth labelling (E11).

Given a total budget of B annotation units - one unit is one class digitised on one tile -
choose which (tile, class) pairs to annotate. A tile carrying two classes costs two units if
both are wanted, because they are two passes and two training samples; it is written out only
under the classes actually bought. The selection is written in the
`cls_selection_*.json` format, so it can be used either as a training set or as a retrieval
pool by passing its name to `make_pool_pairing.py --source`.

Strategies:

    uniform   spend the budget evenly over the subsets, one pair at a time - the n-shot scheme
              used so far, which ignores how redundant a sheet is
    random    B pairs drawn uniformly from the candidate pool
    kcenter   farthest-point sampling, a.k.a. k-center greedy (`fps` is accepted as an
              alias): repeatedly take the tile furthest from everything already chosen, which
              minimises the covering radius - the distance from the worst-served tile in the
              pool to its nearest annotated one. Combined with `--balance`, the walk runs
              inside each cell, so it is every cell's own space that gets covered.
    kmeans    k-means centroids, each replaced by its nearest real tile; trades coverage for
              typicality, and is the guard against k-center chasing outliers

Every strategy spends exactly B units, since the budget is the thing held fixed; how a strategy
distributes them is what it decides.

`--cell-gap` says which map sheets share a retrieval pool, and takes the same value in
`make_pool_pairing`. Each sheet is summarised by the mean of its tiles' embeddings; two sheets
join when that distance is at most the gap, transitively. **0 leaves every sheet on its own**,
which is the validated setting, and this corpus collapses into one pool by 0.114. The budget is
spread over those cells and retrieval draws inside them, so the two must agree - split apart
they can be given annotations retrieval never looks at, or asked for exemplars the budget never
bought.

`--balance` divides the budget over the cells: a quota per (cell, class), per cell, or not at
all. It decides *what* gets covered, and that is not a detail. Run over the whole pool at once,
k-center covers the corpus well, but any single class's picks are then only wherever that class
happens to meet the global cover, which leaves each class's own region patchy and starves a
minority class. `--cell-gap 0 --strategy uniform` is the classic n-per-class-and-sheet scheme.

Candidates are restricted to pairs whose class actually occupies the tile above the
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
from data_tools.indexing.make_pool_pairing import (corpus_map, era_map, group_of,
                                                   style_map,
                                                   fg_threshold, load_samples,
                                                   year_of)

INVALID_CLS = ("background", "nonlabeled")


def kcenter_greedy(vectors, budget, seed, seeds=None):
    """Farthest-point sampling: each step adds the point furthest from the chosen set.

    `seeds` are vectors of tiles the cell already owns. Given them, the walk extends their
    coverage instead of starting a fresh one, so a cell's quota fills the gaps its free tiles
    left rather than re-covering ground that is already served.
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if seeds is not None and len(seeds):
        chosen = []
        # cosine distance to the nearest already-bought tile
        nearest = (1.0 - vectors @ np.asarray(seeds).T).min(axis=1)
    else:
        first = int(rng.randint(n))
        chosen = [first]
        # cosine distance to the nearest chosen point, updated incrementally
        nearest = 1.0 - vectors @ vectors[first]
    while len(chosen) < min(budget, n):
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, 1.0 - vectors @ vectors[pick])
    return chosen


def fps_medoids(vectors, budget, seed, seeds=None, rounds=2):
    """Farthest-point placement, then every pick moved to the centre of what it represents.

    Plain farthest-point sampling buys the periphery: it takes whichever tile is furthest from
    everything chosen, which is by construction the most isolated one. Measured over
    hismap+siegfried at B=300, its picks sit in neighbourhoods 1.8x emptier than the median
    tile and two thirds of them come from the sparsest tenth of the pool - the budget is spent
    on tiles that stand for almost nothing else.

    So the spread is kept and the extremity dropped: assign every tile to its nearest pick, and
    replace each pick with the member closest to that group's mean. The picks stay one per
    well-separated region, but each is now a typical member of its region rather than its most
    eccentric one. This is k-medoids seeded by farthest-point rather than at random, which is
    also why two refinement rounds are enough.
    """
    chosen = kcenter_greedy(vectors, budget, seed, seeds)
    for _ in range(rounds):
        assign = np.argmax(vectors @ vectors[chosen].T, axis=1)
        taken, moved = set(), []
        for j in range(len(chosen)):
            members = np.where(assign == j)[0]
            if not len(members):
                pick = chosen[j]
            else:
                mv = vectors[members]
                centre = mv.mean(axis=0)
                centre /= np.linalg.norm(centre) + 1e-12
                # nearest real tile to the group's mean, skipping any already taken
                for idx in members[np.argsort(-(mv @ centre))]:
                    if int(idx) not in taken:
                        pick = int(idx)
                        break
                else:
                    pick = chosen[j]
            taken.add(int(pick))
            moved.append(int(pick))
        if moved == chosen:
            break
        chosen = moved
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


def load_index(path):
    """Return (vectors, row_of), the vectors L2-normalised so a dot product is a cosine."""
    store = np.load(path, allow_pickle=True)
    names = list(store["names"])
    vectors = store["vectors"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, {n: i for i, n in enumerate(names)}


def build_pool(root_dir, row_of, pool="train_samples", fg_ratio=0.05,
               keywords=None, exclude=None):
    """The tiles a budget may be spent on: {tile: [classes present above the threshold]}.

    Tiles without an annotated class are dropped, so a coverage-seeking strategy cannot spend
    the budget on sheet borders and blank scans. Returns (subset dirs, present_of).
    """
    dirs = subset_dirs(root_dir, keywords, exclude)
    ratios, present_of = {}, {}
    for subset_dir in dirs:
        subset = os.path.basename(subset_dir)
        with open(os.path.join(subset_dir, f"{subset}.json")) as f:
            ratios.update(json.load(f))
        with open(os.path.join(subset_dir, "color_map.json")) as f:
            classes = {c for c in json.load(f) if c not in INVALID_CLS}
        listing = os.path.join(subset_dir, pool)
        listing += ".json" if os.path.isfile(listing + ".json") else ".txt"
        if not os.path.isfile(listing):
            continue
        threshold = fg_threshold(subset, fg_ratio)
        for name in load_samples(listing):
            if name not in row_of:
                continue
            present = [c for c, r in ratios.get(name, {}).items()
                       if c in classes and r > threshold]
            if present:
                present_of[name] = present
    return dirs, present_of


def local_density(group, neighbours=20):
    """Mean cosine distance from each vector to its `neighbours` nearest others in `group`.

    High means the tile sits in empty space relative to the company it is being judged against.
    """
    k = min(neighbours, max(1, len(group) - 1))
    density = np.empty(len(group), dtype=np.float32)
    for start in range(0, len(group), 1000):             # blocked; a 10k x 10k matrix is 470 MB
        d = 1.0 - group[start:start + 1000] @ group.T
        near = np.partition(d, k, axis=1)[:, :k + 1]
        density[start:start + 1000] = np.sort(near, axis=1)[:, 1:k + 1].mean(axis=1)
    return density


def guard_cells(cells, vectors, row_of, percent, neighbours=20, verbose=True):
    """Drop the sparsest `percent` of each cell's candidates, judged inside that cell.

    Farthest-point sampling spends its budget wherever the pool is thinnest, and the thinnest
    parts of a map corpus are its accidents: sheet borders, fold lines, scanning artefacts, the
    lone tile in a style nothing else shares. Over hismap+siegfried at B=300 that is not a
    minor pull - two thirds of the picks come from the sparsest tenth of the pool, in
    neighbourhoods 1.8x emptier than the median tile. Requiring an annotated class removes the
    blank tiles; it does not remove the odd ones.

    Density is ranked *within each cell*, not over the corpus. A global threshold asks whether
    a tile is unusual for the corpus, and then a cell that is small and distinctive - Bayern
    water, 119 tiles in their own corner of the embedding - is uniformly unusual and loses most
    of its candidates, while a large diffuse cell loses none. Ranking inside the cell asks the
    question that was meant: is this tile unusual *for its own class and corpus*. Every cell
    then keeps the same fraction of its candidates, and a tile may stay a candidate for the
    class it is typical of while being dropped for one it is not.
    """
    if not percent:
        return cells
    out, dropped = {}, 0
    for key, pairs in cells.items():
        if len(pairs) < 3:                               # nothing meaningful to rank
            out[key] = pairs
            continue
        density = local_density(vectors[[row_of[t] for t, _ in pairs]], neighbours)
        cut = np.percentile(density, 100 - percent)
        out[key] = [p for i, p in enumerate(pairs) if density[i] <= cut]
        dropped += len(pairs) - len(out[key])
    if verbose:
        total = sum(len(v) for v in cells.values())
        print(f"density guard: dropped the sparsest {percent:g}% of every cell, "
              f"{dropped} of {total} cell memberships")
    return out


def _cell_map(present_of, gap, corpus_of, vectors, row_of, merge_corpora=False):
    """subset -> cell, by union over sheet mean embeddings. gap=0 leaves every sheet alone."""
    by = {}
    for n in present_of:
        by.setdefault(n.rsplit(".", 1)[0], []).append(n)
    return style_map(by, vectors, row_of, corpus_of, gap, merge_corpora)


def cells_of(present_of, gap=0.0, balance="cell-class", corpus_of=None,
             vectors=None, row_of=None, merge_corpora=False):
    """Split the pool into groups of (tile, class) pairs, the unit the budget is spent in.

    A pair is one class annotated on one tile - one digitising pass - and that is what costs
    money. Counting tiles instead would make the second class on an already-chosen tile free,
    which it is not: the training samples it produces are separate samples with their own
    labels. So a tile carrying forest and water offers two pairs and costs two units, and it is
    listed in the output only under the classes actually bought for it.

    `gap` says which sheets share a retrieval pool; `balance` says how the budget is divided
    over them. Every pair lands in exactly one group, so the groups are disjoint and each
    simply buys its quota, with no credit to carry between them.
    """
    cells_by_sheet = _cell_map(present_of, gap, corpus_of, vectors, row_of, merge_corpora)
    cells = {}
    for name, present in present_of.items():
        group = group_of(name.rsplit(".", 1)[0], cells_by_sheet)
        for cls in present:
            key = {"cell-class": (group, cls), "cell": (group,), "none": ("all",)}[balance]
            cells.setdefault(key, []).append((name, cls))
    return {k: sorted(v) for k, v in cells.items()}


def guarded_cells(present_of, vectors, row_of, gap=0.0, balance="cell-class",
                  corpus_of=None, density_guard=0, neighbours=20, verbose=True):
    """The cells a budget is spent over, with each cell's sparsest tail removed."""
    return guard_cells(cells_of(present_of, gap, balance, corpus_of, vectors, row_of),
                       vectors, row_of,
                       density_guard, neighbours, verbose)


def select(present_of, vectors, row_of, budget, strategy, gap=0.0, balance="cell-class",
           corpus_of=None, density_guard=0, min_per_group=1, seed=2026, verbose=True):
    """Spend `budget` annotation units under `strategy`. Returns sorted (tile, class) pairs.

    One unit is one class annotated on one tile - one digitising pass - so a tile carrying two
    classes costs two units if both are bought, and is returned twice. Every strategy spends
    exactly min(budget, available pairs), because the budget is the thing held fixed and a
    strategy that spent less would not be comparable to one that spent it all.

    `balance` chooses what the budget is split evenly over before the strategy is applied
    within each part: nothing, class, corpus, corpus and class, or subset and class. It matters
    more than it looks. Run over the whole pool at once, k-center covers the *corpus* well, but
    a single class's picks are then only wherever that class happens to meet the global cover,
    which leaves each class's own region patchy. Splitting first and sampling inside each cell
    is what makes the selection cover every cell's latent space rather than the union's.
    """
    cells = guarded_cells(present_of, vectors, row_of, gap, balance, corpus_of,
                          density_guard, verbose=verbose)
    available = sum(len(pairs) for pairs in cells.values())     # cells are disjoint in pairs
    budget = min(budget, available)
    rng = np.random.RandomState(seed)
    # `fps` is the same algorithm under the name it is usually given in the sampling literature
    strategy = "kcenter" if strategy == "fps" else strategy

    def choose(pairs, k, bought):
        """Pick k of `pairs` under the chosen strategy, extending the coverage of `bought`.

        A pair is placed in the embedding by its tile, so the geometry is over tiles while the
        thing being bought is a tile-and-class.
        """
        k = min(k, len(pairs))
        if k <= 0:
            return []
        if strategy == "random":
            return [pairs[int(i)] for i in rng.choice(len(pairs), k, replace=False)]
        if strategy == "uniform":
            # one pair at a time, round-robin over the subsets, skipping exhausted ones
            groups = {}
            for p in pairs:
                groups.setdefault(p[0].rsplit(".", 1)[0], []).append(p)
            keys, out, taken, i = sorted(groups), [], set(), 0
            while len(out) < k:
                g = keys[i % len(keys)]
                i += 1
                groups[g] = left = [m for m in groups[g] if m not in taken]
                if left:
                    pick = left[int(rng.randint(len(left)))]
                    out.append(pick)
                    taken.add(pick)
            return out
        sub = vectors[[row_of[t] for t, _ in pairs]]
        seed_vecs = vectors[[row_of[t] for t, _ in bought]] if bought else None
        if strategy == "kcenter":
            idx = kcenter_greedy(sub, k, seed, seed_vecs)
        elif strategy == "fps-medoid":
            idx = fps_medoids(sub, k, seed, seed_vecs)
        else:
            idx = kmeans_medoids(sub, k, seed)
        return [pairs[int(i)] for i in idx]

    # Equal quotas per cell, smallest cell first so a cell that cannot fill its share hands the
    # shortfall on while the others still have room. Cells are disjoint, so a cell simply buys
    # its quota outright - nothing it buys can satisfy another cell's.
    order = sorted(cells, key=lambda k: (len(cells[k]), k))
    quota = {k: budget // len(cells) for k in cells}
    for i in range(budget % len(cells)):
        quota[order[i]] += 1
    if verbose and balance != "none":
        shown = ", ".join(f"{'/'.join(k)}={quota[k]}" for k in order[:12])
        print(f"{len(cells)} cells: {shown}" + (", ..." if len(cells) > 12 else ""))

    # A floor per (subset, class), spent before the strategy gets the rest. Retrieval is
    # subset-local, so a combination the selection misses entirely has no exemplar in this run's
    # own pool: make_pool_pairing then writes no entry for it, and cd_syntra2 builds its sample
    # list *from the pairing file*, so every test target of that class in that sheet silently
    # leaves the evaluation. Two runs would be scored on different test sets, and skipping a
    # hard class would look like doing well at it. One pair per combination is the correctness
    # floor; two additionally makes each of them trainable, because a target is excluded from
    # its own source pool and a lone tile can prompt nothing.
    selected = []
    if min_per_group:
        # The floor groups by whatever the retrieval scope will be, because that is what
        # decides which combinations need an exemplar of their own. Balancing by era and then
        # retrieving per sheet needs a floor per sheet; matching the two needs only a floor
        # per era, which over this corpus is 43 combinations rather than 88.
        cells_by_sheet = _cell_map(present_of, gap, corpus_of, vectors, row_of)
        scope_of = lambda sub: group_of(sub, cells_by_sheet)
        groups = {}
        for pairs in cells.values():
            for tile, cls in pairs:
                groups.setdefault((scope_of(tile.rsplit(".", 1)[0]), cls), []).append((tile, cls))
        # The density guard cuts per (corpus, class) and can empty a (subset, class) subgroup
        # outright, which would lose the very combination the floor exists to protect. Coverage
        # wins: for those, and only those, fall back to the unguarded candidates.
        every = {}
        for tile, present in present_of.items():
            for cls in present:
                every.setdefault((scope_of(tile.rsplit(".", 1)[0]), cls), []).append((tile, cls))
        for key in sorted(every):
            selected += choose(sorted(groups.get(key) or every[key]), min_per_group, [])
        if verbose:
            print(f"floor: {len(selected)} pairs reserved, {min_per_group} per "
                  f"(cell, class) over {len(every)} combinations")
        if len(selected) > budget:
            raise SystemExit(f"budget {budget} cannot cover {len(every)} (cell, class) "
                             f"combinations at {min_per_group} each; raise it to "
                             f"{len(selected)} or lower --min-per-cell-class")

    for key in order:
        taken = set(selected)
        have = sum(1 for p in cells[key] if p in taken)
        # capped, because a floor pick taken from the unguarded fallback belongs to no guarded
        # cell and so is invisible to every `have` - without the cap those pairs are paid for
        # twice and the run overshoots its budget
        need = min(quota[key] - have, budget - len(selected))
        selected += choose([p for p in cells[key] if p not in taken], need, [])

    # A cell smaller than its quota leaves the budget short. Hand the remainder back to the
    # cells that still have candidates, one pair at a time, rather than to a walk over the
    # whole pool - that would undo the balancing it was meant to keep.
    while len(selected) < budget:
        taken = set(selected)
        spent = len(selected)
        for key in order:
            if len(selected) >= budget:
                break
            mine = [p for p in cells[key] if p in taken]
            fresh = [p for p in cells[key] if p not in taken]
            got = choose(fresh, 1, mine)
            selected += got
            taken.update(got)
        if len(selected) == spent:              # every cell exhausted
            break

    selected = sorted(set(selected))
    assert len(selected) == budget, f"spent {len(selected)} of {budget}"
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--budget", type=int, required=True,
                        help="annotation units to spend, one unit = one class on one tile; "
                             "every strategy spends exactly this many")
    parser.add_argument("--strategy", default="kcenter",
                        choices=["uniform", "random", "kcenter", "fps", "fps-medoid", "kmeans"])
    parser.add_argument("--cell-gap", type=float, default=0.0,
                        help="sheets whose mean embeddings lie within this cosine gap share a "
                             "retrieval pool. 0 is sheet-wise, the validated setting; this "
                             "corpus collapses to one pool by 0.114. Pass the same value to "
                             "make_pool_pairing")
    parser.add_argument("--balance", default="cell-class",
                        choices=["cell-class", "cell", "none"],
                        help="how the budget is divided over the cells: a quota per (cell, "
                             "class), per cell, or not at all")
    parser.add_argument("--pool", default="train_samples",
                        help="sample list the budget may be spent from")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--fg-ratio", default="0.05",
                        help="float, or key=value pairs, e.g. hameln=0.05,siegfried=0.01")
    parser.add_argument("--density-guard", type=float, default=0,
                        help="drop the sparsest N%% of each cell from candidacy first; 10 "
                             "stops farthest-point sampling spending the budget on outliers")
    parser.add_argument("--min-per-cell-class", type=int, default=1,
                        help="floor of pairs per (cell, class), spent before the strategy. "
                             "1 keeps every run's evaluation set identical; 2 also makes every "
                             "bought pair trainable, at a much larger share of a small budget")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="hold this fraction of the budget out as validation, written to "
                             "<name>_val.json; it is paid for out of the budget, not added to it")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default=None,
                        help="output name; default budget_<strategy>[_<balance>]_<B>")
    parser.add_argument("--output-root", default=None,
                        help="write into a parallel tree instead of the dataset itself")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fg_ratio = ({k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
                if "=" in args.fg_ratio else float(args.fg_ratio))

    vectors, row_of = load_index(args.index)
    dirs, present_of = build_pool(args.root_dir, row_of, args.pool, fg_ratio,
                                  args.data_keywords, args.exclude_keys)

    all_classes = sorted({c for present in present_of.values() for c in present})
    candidates = sum(len(p) for p in present_of.values())
    print(f"candidate pool: {candidates} (tile, class) pairs over {len(present_of)} annotated "
          f"tiles in {len(dirs)} subsets")

    selected = select(present_of, vectors, row_of, args.budget, args.strategy,
                      gap=args.cell_gap, balance=args.balance, corpus_of=corpus_map(dirs),
                      density_guard=args.density_guard,
                      min_per_group=args.min_per_cell_class, seed=args.seed)

    # The validation split comes out of the budget, not from extra labels: choosing the
    # iteration count needs a held-out signal, and borrowing an annotated set from outside
    # would understate the annotation cost, which is the one number this is all about. Split
    # per class so every class is represented on both sides.
    val_pairs = []
    if args.val_fraction:
        rng = np.random.RandomState(args.seed)
        by_cls = {}
        for pair in selected:
            by_cls.setdefault(pair[1], []).append(pair)
        for cls, group in sorted(by_cls.items()):
            take = int(round(args.val_fraction * len(group)))
            if 0 < take < len(group):
                idx = rng.choice(len(group), take, replace=False)
                val_pairs += [group[int(i)] for i in idx]
        val_pairs = sorted(set(val_pairs))
        selected = [p for p in selected if p not in set(val_pairs)]
        print(f"validation split: {len(val_pairs)} of {len(val_pairs) + len(selected)} pairs "
              f"held out, {len(selected)} left to train on")

    # Only the classes actually bought for a tile are written. A class the budget did not pay
    # for is not annotated, so listing it would hand the training set labels that do not exist.
    by_class = {}
    for tile, cls in selected:
        by_class.setdefault(cls, []).append(tile)
    val_by_class = {}
    for tile, cls in val_pairs:
        val_by_class.setdefault(cls, []).append(tile)

    tiles = {t for t, _ in selected}
    subsets_hit = {t.rsplit(".", 1)[0] for t in tiles}
    print(f"strategy={args.strategy}"
          f" cell-gap={args.cell_gap:g} balance={args.balance} "
          f"budget={len(selected)}: {len(selected)} pairs over {len(tiles)} tiles, "
          f"{len(subsets_hit)}/{len(dirs)} subsets covered")
    for cls in all_classes:
        print(f"    {cls:<14} {len(by_class.get(cls, [])):>5} pairs")

    name = args.name or (f"budget_{args.strategy}"
                         f"{'' if not args.density_guard else f'_g{args.density_guard:g}'}"
                         f"_gap{args.cell_gap:g}"
                         f"_{args.budget}")
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
        for suffix, table in (("", by_class), ("_val", val_by_class)):
            if suffix and not val_pairs:
                continue
            own = {c: sorted(n for n in v if n.rsplit(".", 1)[0] == subset)
                   for c, v in table.items()}
            own = {c: v for c, v in own.items() if v}
            with open(os.path.join(out_dir, f"{name}{suffix}.json"), "w") as f:
                json.dump(own, f, indent=2)
    print(f"\nWrote {name}.json to {len(dirs)} subsets")


if __name__ == "__main__":
    main()
