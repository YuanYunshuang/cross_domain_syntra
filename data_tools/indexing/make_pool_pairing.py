# -*- coding: utf-8 -*-
"""Generate retrieval pairings at a configurable scope: within a sheet, a footprint, or globally.

For every target tile and every class, the k nearest annotated tiles are looked up in the
embedding store built by `build_global_index.py` and written in the `sample_pair_*` format that
the dataset already reads:

    {target: {class: {"src_samples": [...], "distances": [...]}}}

The scope decides which annotated tiles may serve as sources:

    subset     only tiles of the same map sheet and year - what all runs so far used, and a
               setting in which the style is already homogeneous
    footprint  all years of the same geographic footprint, e.g. every hameln.3921_*
    global     every annotated tile in the store, across sheets and years

No dataset change is needed for cross-subset sources: the loader resolves each source name to
its own subset folder.

    PYTHONPATH=. python data_tools/indexing/make_pool_pairing.py \\
        -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \\
        --source train_samples --target test_samples --scope global -k 8 \\
        -dkw hismap -xkw hameln.3921_2023
"""

import argparse
import json
import os

import numpy as np

from data_tools.indexing.build_global_index import subset_dirs

INVALID_CLS = ("background", "nonlabeled")


def footprint_of(subset):
    """`hameln.3921_2017` -> `hameln.3921`; subsets without a year keep their own name."""
    if "." in subset and "_" in subset.rsplit(".", 1)[-1]:
        head, tail = subset.rsplit(".", 1)
        return f"{head}.{tail.split('_')[0]}"
    return subset


def load_classes(path):
    """{tile: [classes it was annotated for]} for a selection json, or None for a plain list.

    A budgeted selection is a set of (tile, class) pairs, not a set of tiles: a tile bought for
    water was not annotated for forest. Flattening it to tiles and re-deriving the classes from
    the label ratios - which is what happens without this - hands back exactly the classes the
    budget declined to buy: on the source side as prompts with no annotation behind them, on the
    target side as training samples with ground truth nobody paid for. A `*_samples.txt` list
    carries no class restriction and returns None, which leaves the old behaviour intact.
    """
    if not path.endswith(".json"):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for cls, names in data.items():
        for name in names:
            out.setdefault(name, []).append(cls)
    return out


def load_samples(path):
    """Sample names from a `*_samples.txt` or a `cls_selection_<n>.json`."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        names = []
        for values in data.values():
            names.extend(values)
        return sorted(set(names))
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def mirror_subset(source_dir, output_root, subset, data_root):
    """Recreate one subset under `output_root` as symlinks to the original content.

    Only the generated pairing files are written as real files there, so the dataset directory
    that backs the published release is never modified. `imgs/` and `lbls/` are linked as whole
    directories; the small metadata and sample lists are linked individually.

    The path relative to the data root is preserved (`hismap/hameln.3922_1898`, not
    `hameln.3922_1898`), because both `data_keywords` and the per-dataset `foreground_ratio`
    match on substrings of the subset path.
    """
    relative = os.path.relpath(source_dir, data_root)
    target_dir = os.path.join(output_root, relative)
    os.makedirs(target_dir, exist_ok=True)
    for entry in sorted(os.listdir(source_dir)):
        if entry.startswith("sample_pair_") or entry.startswith("pairing_"):
            continue  # pairings are what this tool produces; do not inherit stale ones
        link = os.path.join(target_dir, entry)
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            continue  # a real file someone put there on purpose
        os.symlink(os.path.join(source_dir, entry), link)
    return target_dir


def fg_threshold(subset, fg_ratio):
    if isinstance(fg_ratio, (int, float)):
        return float(fg_ratio)
    for key, value in fg_ratio.items():
        if key in subset:
            return float(value)
    return 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--index", required=True, help="npz written by build_global_index.py")
    parser.add_argument("--source", default="train_samples",
                        help="source sample list per subset, without extension")
    parser.add_argument("--target", default="test_samples", help="target sample list per subset")
    parser.add_argument("--scope", default="global", choices=["subset", "footprint", "global"])
    parser.add_argument("-k", type=int, default=8, help="neighbours to record per target and class")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--fg-ratio", default="0.05",
                        help="float, or comma-separated key=value pairs, e.g. hameln=0.05,siegfried=0.01")
    parser.add_argument("--candidates", default="class", choices=["class", "style"],
                        help="which annotated tiles may serve as sources for a class. 'class' "
                             "keeps tiles that contain that class above the foreground "
                             "threshold. 'style' is reserved for pooling by visual style "
                             "regardless of which class a tile carries - the way a cartographer "
                             "would first pick a similar sheet and only then its annotations.")
    parser.add_argument("--suffix", default=None,
                        help="output file suffix; defaults to the scope name")
    parser.add_argument("--output-root", default=None,
                        help="write into a parallel tree instead of into the dataset itself. "
                             "Each subset is recreated there with symlinks to the original "
                             "imgs/, lbls/ and metadata, so the released data stays untouched "
                             "while the loader still finds everything beside the pairing file. "
                             "Point root_dir at this tree to use the generated pools.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.candidates == "style":
        raise NotImplementedError(
            "Style-scoped candidate pools are not implemented yet. The hook exists because it "
            "changes what a pool is: sources would be chosen by visual similarity alone and "
            "their class annotations taken as given, rather than filtering to tiles that "
            "already contain the queried class.")

    if "=" in args.fg_ratio:
        fg_ratio = {k.split("=")[0]: float(k.split("=")[1]) for k in args.fg_ratio.split(",")}
    else:
        fg_ratio = float(args.fg_ratio)

    store = np.load(args.index, allow_pickle=True)
    names = list(store["names"])
    vectors = store["vectors"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    row_of = {n: i for i, n in enumerate(names)}
    print(f"index: {len(names)} embeddings")

    dirs = subset_dirs(args.root_dir, args.data_keywords, args.exclude_keys)
    subsets = [os.path.basename(d) for d in dirs]
    print(f"{len(dirs)} subsets, scope={args.scope}, k={args.k}")

    # class ratios and the source pool of every subset
    ratios, sources, source_classes = {}, {}, {}
    for subset_dir, subset in zip(dirs, subsets):
        with open(os.path.join(subset_dir, f"{subset}.json")) as f:
            ratios.update(json.load(f))
        source_path = os.path.join(subset_dir, args.source)
        source_path += ".json" if os.path.isfile(source_path + ".json") else ".txt"
        if not os.path.isfile(source_path):
            print(f"  !! {subset}: no source list {args.source}")
            continue
        sources[subset] = [n for n in load_samples(source_path) if n in row_of]
        source_classes[subset] = load_classes(source_path)

    # candidate tiles per (scope key, class)
    candidates, unpaid = {}, 0
    for subset, pool in sources.items():
        keys = {"subset": [subset], "footprint": [footprint_of(subset)], "global": ["global"]}[args.scope]
        threshold = fg_threshold(subset, fg_ratio)
        bought = source_classes.get(subset)
        for name in pool:
            allowed = None if bought is None else set(bought.get(name, ()))
            for cls, ratio in ratios.get(name, {}).items():
                if cls in INVALID_CLS or ratio <= threshold:
                    continue
                if allowed is not None and cls not in allowed:
                    unpaid += 1              # present in the tile, but not annotated for it
                    continue
                for key in keys:
                    candidates.setdefault((key, cls), []).append(name)
    if unpaid:
        print(f"class restriction: {unpaid} (tile, class) combinations skipped as sources - "
              f"the class is in the tile but the budget did not buy it")
    print(f"candidate pools: {len(candidates)} (scope key, class) combinations, "
          f"largest {max((len(v) for v in candidates.values()), default=0)} tiles")

    suffix = args.suffix or args.scope
    for subset_dir, subset in zip(dirs, subsets):
        target_path = os.path.join(subset_dir, args.target)
        target_path += ".json" if os.path.isfile(target_path + ".json") else ".txt"
        if not os.path.isfile(target_path):
            print(f"  !! {subset}: no target list {args.target}")
            continue
        targets = [n for n in load_samples(target_path) if n in row_of]
        target_classes = load_classes(target_path)
        scope_key = {"subset": subset, "footprint": footprint_of(subset), "global": "global"}[args.scope]

        # Only classes the target subset is annotated for. A wider pool offers sources for
        # classes this sheet never annotated - grass for a railway tile, settlement for a
        # Donauwörth sheet - but those pairs have no ground truth: an unannotated class is not
        # an absent one, so they can be neither trained on nor scored.
        with open(os.path.join(subset_dir, "color_map.json")) as f:
            own_classes = {c for c in json.load(f) if c not in INVALID_CLS}
        classes = sorted({c for (key, c) in candidates if key == scope_key} & own_classes)
        pairing = {t: {} for t in targets}
        query = vectors[[row_of[t] for t in targets]]

        for cls in classes:
            pool = candidates[(scope_key, cls)]
            pool_rows = np.array([row_of[n] for n in pool])
            sims = query @ vectors[pool_rows].T                    # (T, C), cosine
            for i, target in enumerate(targets):
                # a budgeted target is only a training sample for the classes it was bought for
                if target_classes is not None and cls not in target_classes.get(target, ()):
                    continue
                row = sims[i]
                if target in pool:                                  # never prompt with the target
                    row = row.copy()
                    row[pool.index(target)] = -np.inf
                take = min(args.k, int(np.isfinite(row).sum()))
                if take <= 0:
                    continue
                top = np.argpartition(-row, take - 1)[:take]
                top = top[np.argsort(-row[top])]
                pairing[target][cls] = {
                    "src_samples": [pool[j] for j in top],
                    "distances": [round(float(1 - row[j]), 6) for j in top],
                }

        out_dir = subset_dir
        if args.output_root and not args.dry_run:
            out_dir = mirror_subset(subset_dir, args.output_root, subset, args.root_dir)
        out = os.path.join(out_dir, f"sample_pair_{args.source}_{args.target}_{suffix}.json")
        n_pairs = sum(len(v) for v in pairing.values())
        cross = sum(1 for v in pairing.values() for c in v.values()
                    for s in c["src_samples"] if s.rsplit(".", 1)[0] != subset)
        total = sum(len(c["src_samples"]) for v in pairing.values() for c in v.values())
        print(f"  {subset:<24} {len(targets):>5} targets, {n_pairs:>6} target-class pairs, "
              f"{100 * cross / max(total, 1):>5.1f}% sources from other subsets")
        if not args.dry_run:
            with open(out, "w") as f:
                json.dump(pairing, f)

    print("\nDone." + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
