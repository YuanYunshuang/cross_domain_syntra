# -*- coding: utf-8 -*-
"""Union several selections into one, so arms can share a component exactly.

Experiment 1 varies how positives are chosen and holds everything else still. Negatives are part
of "everything else": if each arm drew its own, a difference in the result would not say whether
the positives or the negatives produced it. Drawing them once and merging the same set into both
arms removes the question - the two selections then differ in their positive half and in nothing
else at all.

    PYTHONPATH=. python data_tools/indexing/merge_selections.py \\
        -d /home/yuan/data/syntra_e13 --inputs e13x1_fps_pos,e13x1_neg --name e13x1_fps \\
        -dkw hismap,siegfried -xkw hameln.3921_2023
"""

import argparse
import json
import os

from data_tools.indexing.build_global_index import subset_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--root-dir", required=True)
    parser.add_argument("--inputs", required=True, help="comma-separated selection names")
    parser.add_argument("--name", required=True, help="output name")
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    args = parser.parse_args()

    names = [n.strip() for n in args.inputs.split(",") if n.strip()]
    total, overlap = 0, 0
    for subset_dir in subset_dirs(args.root_dir, args.data_keywords, args.exclude_keys):
        merged = {}
        for name in names:
            path = os.path.join(subset_dir, f"{name}.json")
            if not os.path.isfile(path):
                continue
            for cls, tiles in json.load(open(path)).items():
                seen = merged.setdefault(cls, set())
                overlap += len(seen & set(tiles))
                seen.update(tiles)
        if not merged:
            continue
        with open(os.path.join(subset_dir, f"{args.name}.json"), "w") as f:
            json.dump({c: sorted(v) for c, v in sorted(merged.items())}, f, indent=2)
        total += sum(len(v) for v in merged.values())
    print(f"{args.name}: {total} pairs from {' + '.join(names)}"
          + (f"   ({overlap} appeared in more than one and were counted once)" if overlap else ""))


if __name__ == "__main__":
    main()
