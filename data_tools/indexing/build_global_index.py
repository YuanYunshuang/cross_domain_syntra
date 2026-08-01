# -*- coding: utf-8 -*-
"""Merge the per-subset DINOv2 indices into one embedding store, and fill in missing tiles.

Each subset carries `<split>.faiss` (an `IndexFlatL2` of DINOv2 ViT-S/14 CLS tokens) plus a
`<split>_map.json` listing the image path of every row. This tool reconstructs those vectors
into a single `.npz` keyed by `<subset>.<tile>`, which is what cross-subset retrieval and the
budget selectors work on. Tiles that were never indexed - the test split, for instance - are
embedded on the fly with the same encoder and the same preprocessing, so old and new vectors
are comparable.

    # merge what exists, and embed the test tiles that do not
    PYTHONPATH=. python data_tools/indexing/build_global_index.py \\
        -d /home/yuan/data/syntra --splits train_samples,test_samples \\
        -dkw hismap,siegfried -o /home/yuan/data/syntra/global_index.npz
"""

import argparse
import json
import os
from glob import glob

import numpy as np

DIM = 384  # DINOv2 ViT-S/14 CLS token


def subset_dirs(root, keywords=None, exclude=None):
    dirs = sorted(os.path.dirname(p) for p in
                  glob(os.path.join(root, "**", "color_map.json"), recursive=True))
    if keywords:
        keys = [k.strip() for k in keywords.split(",")]
        dirs = [d for d in dirs if any(k in d for k in keys)]
    if exclude:
        keys = [k.strip() for k in exclude.split(",")]
        dirs = [d for d in dirs if not any(k in d for k in keys)]
    return dirs


def load_subset_index(subset_dir, split):
    """Return {name: vector} from an existing faiss index, or {} if there is none."""
    index_path = os.path.join(subset_dir, f"{split}.faiss")
    map_path = os.path.join(subset_dir, f"{split}_map.json")
    if not (os.path.isfile(index_path) and os.path.isfile(map_path)):
        return {}

    import faiss
    index = faiss.read_index(index_path)
    with open(map_path) as f:
        paths = json.load(f)
    if index.ntotal != len(paths):
        print(f"    !! {os.path.basename(subset_dir)}/{split}: index has {index.ntotal} rows "
              f"but the map lists {len(paths)}; skipping")
        return {}

    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    subset = os.path.basename(subset_dir)
    return {f"{subset}.{os.path.splitext(os.path.basename(p))[0]}": v
            for p, v in zip(paths, vectors)}


def wanted_names(subset_dir, split):
    """Sample names listed in `<split>.txt`, or all tiles if that file does not exist."""
    listing = os.path.join(subset_dir, f"{split}.txt")
    subset = os.path.basename(subset_dir)
    if os.path.isfile(listing):
        with open(listing) as f:
            return [line.strip() for line in f if line.strip()]
    return [f"{subset}.{os.path.splitext(os.path.basename(p))[0]}"
            for p in sorted(glob(os.path.join(subset_dir, "imgs", "*")))]


def embed_missing(names, dir_of_subset, batch_size, num_workers):
    """Embed tiles that no index covers, with the encoder the stored vectors came from.

    `BaseDataloaders` assumes every subset sits directly under the data root; here they are
    nested (``hismap/hameln.3922_1898``, ``siegfried/railway``), so the dataset is built per
    subset from the directory that was actually discovered.
    """
    from torch.utils.data import DataLoader
    from data_tools.indexing.indexdb import BaseDataset, DinoV2Encoder

    by_subset = {}
    for name in names:
        subset, tile = name.rsplit(".", 1)
        by_subset.setdefault(subset, []).append(tile)

    encoder = DinoV2Encoder()
    out = {}
    for subset, tiles in sorted(by_subset.items()):
        subset_dir = dir_of_subset.get(subset)
        if subset_dir is None:
            print(f"    !! no directory found for {subset}, skipping {len(tiles)} tiles")
            continue
        dataset = BaseDataset(subset_dir, sorted(tiles))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=dataset.collate_fn)
        for paths, images in loader:
            emb = encoder.generate_embedding(images.cuda())["x_norm_clstoken"]
            for p, v in zip(paths, emb.cpu().numpy().astype(np.float32)):
                out[f"{subset}.{os.path.splitext(os.path.basename(p))[0]}"] = v
        print(f"    embedded {subset:<24} {len(tiles):>6} tiles")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--splits", default="train_samples",
                        help="comma-separated splits to cover, e.g. train_samples,test_samples")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--compute-missing", action="store_true",
                        help="embed tiles that no index covers (needs a GPU and DINOv2)")
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("-nw", "--num-workers", type=int, default=8)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    dirs = subset_dirs(args.root_dir, args.data_keywords, args.exclude_keys)
    splits = [s.strip() for s in args.splits.split(",")]
    print(f"{len(dirs)} subsets, splits {splits}")

    dir_of_subset = {os.path.basename(d): d for d in dirs}
    store, missing = {}, []
    for subset_dir in dirs:
        found = 0
        for split in splits:
            vectors = load_subset_index(subset_dir, split)
            store.update(vectors)
            found += len(vectors)
            absent = [n for n in wanted_names(subset_dir, split) if n not in store]
            missing.extend(absent)
        print(f"  {os.path.basename(subset_dir):<24} {found:>5} indexed")

    missing = sorted(set(missing) - set(store))
    if missing:
        print(f"\n{len(missing)} tiles are not covered by any index")
        if args.compute_missing:
            store.update(embed_missing(missing, dir_of_subset, args.batch_size, args.num_workers))
        else:
            print("  pass --compute-missing to embed them")

    names = sorted(store)
    vectors = np.stack([store[n] for n in names]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, names=np.array(names), vectors=vectors)
    print(f"\nWrote {len(names)} embeddings of dim {vectors.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
