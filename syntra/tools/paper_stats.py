# -*- coding: utf-8 -*-
"""Regenerate every corpus- and noise-level number the paper quotes, into results/paper_stats.csv.

These numbers were first computed ad hoc and recorded only in the docs and commit messages. This
script is their source. Each row gives the value, how it is defined, and the inputs it was read
from, so a reviewer's "where does this come from" has a one-command answer:

    python syntra/tools/paper_stats.py            # CPU only, reads labels and result files

Five groups:

  corpus    tiles, (tile, class) pairs, positives, negatives, pairs per tile, natural neg:pos
            ratio. A pair is one tile and one class in its subset's vocabulary. It is
            positive when the class covers more than 5% of the tile (the flat foreground_ratio).
  coreg     whether one footprint's survey years are the same ground: correlation of the
            per-cell forest fraction between two years of one footprint, against the same
            statistic between two different footprints matched by cell name (the null).
  thin      per-class positive mass against negative exposure on the test set, and the
            correlation of S3's seed spread with log(mass / neg). This is why the pooled IoU of
            thin classes is fragile.
  evalnoise the same checkpoint scored twice (the ablation's own test and the num_src grid at its
            training k), against the spread across training seeds.
"""

import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = "/home/yuan/data/syntra"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = "syntra/tools/paper_stats.py"
FG = 0.05
INVALID = ("background", "nonlabeled")
RESULT = "test_model_latest_hismap,siegfried.txt"

rows = []


def add(group, stat, value, definition, inputs):
    rows.append((group, stat, value, definition, inputs, SCRIPT))


def subsets():
    """{subset: folder}. hismap sheets plus the two Siegfried sets, as the paper's corpus."""
    dirs = sorted(os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/color_map.json", recursive=True))
    dirs = [d for d in dirs if "/hismap/" in d or d.endswith(("siegfried/railway", "siegfried/vineyard"))]
    return {os.path.basename(d): d for d in dirs}


def corpus_of(subset):
    return subset.split(".")[0] if "." in subset else subset


def load(subset, folder):
    with open(f"{folder}/{subset}.json") as f:
        ratios = json.load(f)
    with open(f"{folder}/color_map.json") as f:
        vocab = [c for c in json.load(f) if c not in INVALID]
    return ratios, vocab


def samples(folder, split):
    with open(f"{folder}/{split}_samples.txt") as f:
        return [line.strip() for line in f if line.strip()]


def parse_result(path):
    """{class: IoU} plus 'mIoU', from the last block of a test result file."""
    text = open(path).read()
    block = text.split("Evaluation Results")[-1]
    out = {m.group(1): float(m.group(2)) for m in
           re.finditer(r"Notion set : (\S+): \s*\n\s*IoU \(FG\): ([\d.]+)", block)}
    mean = re.search(r"Overal Mean #+\s*\n\s*IoU \(FG\): ([\d.]+)", block)
    out["mIoU"] = float(mean.group(1))
    return out


def corpus_stats(subs):
    total = {"tiles": 0, "pairs": 0, "pos": 0, "neg": 0}
    per = {}
    for s, d in subs.items():
        ratios, vocab = load(s, d)
        c = per.setdefault(corpus_of(s), {"subsets": 0, "tiles": 0, "pairs": 0, "pos": 0})
        c["subsets"] += 1
        for r in ratios.values():
            c["tiles"] += 1
            c["pairs"] += len(vocab)
            c["pos"] += sum(r.get(k, 0.0) > FG for k in vocab)
    src = "<subset>/<subset>.json, color_map.json for every hismap subset + siegfried railway, vineyard"
    for name, c in sorted(per.items()):
        add("corpus", f"{name}: subsets / tiles / pairs / pairs per tile",
            f"{c['subsets']} / {c['tiles']} / {c['pairs']} / {c['pairs'] / c['tiles']:.1f}",
            "pairs = tiles x classes in the subset vocabulary", src)
        for k in total:
            total[k] += c.get(k, 0)
    total["subsets"] = sum(c["subsets"] for c in per.values())
    total["neg"] = total["pairs"] - total["pos"]
    add("corpus", "subsets", total["subsets"], "hismap sheets + railway + vineyard", src)
    add("corpus", "tiles", total["tiles"], "every tile in the corpus, train and test", src)
    add("corpus", "(tile, class) pairs", total["pairs"], "tile x class in its subset's vocabulary", src)
    add("corpus", "positive pairs", total["pos"], f"class covers > {FG:g} of the tile", src)
    add("corpus", "negative pairs", total["neg"], "pairs minus positives", src)
    add("corpus", "pairs per tile", f"{total['pairs'] / total['tiles']:.2f}", "pairs / tiles", src)
    add("corpus", "natural neg:pos ratio", f"{total['neg'] / total['pos']:.2f}",
        "negative pairs / positive pairs, if every tile is fully labelled", src)
    tr = sum(len(samples(d, "train")) for d in subs.values())
    te = sum(len(samples(d, "test")) for d in subs.values())
    add("corpus", "train / test tiles", f"{tr} / {te}", "train_samples.txt, test_samples.txt",
        "<subset>/{train,test}_samples.txt")


def coreg_stats(subs):
    hameln = {s: d for s, d in subs.items() if s.startswith("hameln.")}
    cell = {s: {k.rsplit(".", 1)[1]: v.get("forest", 0.0) for k, v in load(s, d)[0].items()}
            for s, d in hameln.items()}
    footprint = lambda s: s.split("_")[0]
    year = lambda s: int(s.split("_")[1])

    def r(a, b):
        keys = sorted(set(cell[a]) & set(cell[b]))
        return np.corrcoef([cell[a][k] for k in keys], [cell[b][k] for k in keys])[0, 1], len(keys)

    same, cross = [], []
    for a, b in itertools.combinations(sorted(hameln), 2):
        (same if footprint(a) == footprint(b) else cross).append((a, b, *r(a, b)))
    src = "hameln.*/hameln.*.json, forest fraction per cell (tile name minus subset)"
    grids = {footprint(s): len(cell[s]) for s in cell}
    shared = all(set(cell[a]) == set(cell[b]) for a, b, *_ in same)
    add("coreg", "cells per footprint grid", " / ".join(f"{k} {v}" for k, v in sorted(grids.items())),
        "tiles per subset; every year of a footprint shares the same cell names" if shared else
        "tiles per subset; WARNING some years of a footprint do not share cell names", src)
    v = [x[2] for x in same]
    gaps = [abs(year(a) - year(b)) for a, b, *_ in same]
    add("coreg", "same footprint, all year pairs: r min / median / max",
        f"{min(v):.3f} / {np.median(v):.3f} / {max(v):.3f}",
        f"Pearson r of forest fraction per cell, {len(same)} pairs, gaps {min(gaps)}-{max(gaps)} y", src)
    held = [(b if a == "hameln.3922_1996" else a, rr) for a, b, rr, _ in same
            if "hameln.3922_1996" in (a, b)]
    add("coreg", "held-out 1996 vs its footprint's other years: r min / max",
        f"{min(x[1] for x in held):.3f} / {max(x[1] for x in held):.3f}",
        "; ".join(f"{s.split('_')[1]} {x:.3f}" for s, x in sorted(held)), src)
    v = [x[2] for x in cross]
    add("coreg", "different footprints: r min / median / max",
        f"{min(v):.3f} / {np.median(v):.3f} / {max(v):.3f}",
        f"same statistic, {len(cross)} cross-footprint pairs matched by cell name, "
        f"{min(x[3] for x in cross)} shared cells (grids differ in extent)", src)
    nulls = [(b if a == "hameln.3922_1996" else a, rr) for a, b, rr, _ in cross
             if "hameln.3922_1996" in (a, b)]
    add("coreg", "held-out 1996 vs other footprints: r min / max",
        f"{min(x[1] for x in nulls):.3f} / {max(x[1] for x in nulls):.3f}",
        "the best case is 3821 (~0.27); most other footprints are negative", src)


SEEDS = {"S3": ["S3_k3_retrieved", "S3_k3_retrieved_s1235", "S3_k3_retrieved_s1236"],
         "S0": ["S0_smol_k1", "S0_smol_k1_s1235", "ratio_smol_p0.5"]}


def thin_stats(subs):
    """Per class: positive mass (sum of coverage, in tiles) and negative test tiles."""
    mass, neg, zero, n = {}, {}, {}, {}
    for s, d in subs.items():
        ratios, vocab = load(s, d)
        for t in samples(d, "test"):
            for k in vocab:
                key = f"{corpus_of(s)}.{k}"
                v = ratios[t].get(k, 0.0)
                n[key] = n.get(key, 0) + 1
                mass[key] = mass.get(key, 0.0) + v * (v > FG)
                neg[key] = neg.get(key, 0) + (v <= FG)
                zero[key] = zero.get(key, 0) + (v == 0.0)
    runs = [parse_result(f"{REPO}/logs/{r}/{RESULT}") for r in SEEDS["S3"]]
    classes = sorted(k for k in runs[0] if k != "mIoU")
    sd = {c: float(np.std([r[c] for r in runs], ddof=1)) for c in classes}
    src_lbl = "<subset>/<subset>.json + test_samples.txt"
    for c in classes:
        add("thin", f"{c}: test tiles / pos mass / neg tiles (<=5%) / tiles with none / S3 seed sd",
            f"{n[c]} / {mass[c]:.0f} / {neg[c]} / {zero[c]} / {sd[c]:.2f}",
            "pos mass = sum of class coverage over positive (>5%) test tiles, in tile units",
            f"{src_lbl}; logs/{{{','.join(SEEDS['S3'])}}}/{RESULT}")
    x = [np.log(mass[c] / neg[c]) for c in classes]
    y = [sd[c] for c in classes]
    add("thin", "r(S3 seed sd, log(pos mass / neg tiles)) over the nine classes",
        f"{np.corrcoef(x, y)[0, 1]:.2f}", "neg tiles = test tiles with coverage <= 5%; sd with ddof=1",
        f"{src_lbl}; logs/S3_k3_retrieved*/{RESULT}")
    x0 = [np.log(mass[c] / max(zero[c], 1)) for c in classes]
    add("thin", "same, with neg = tiles with zero coverage",
        f"{np.corrcoef(x0, y)[0, 1]:.2f}", "sensitivity check on the definition of a negative",
        f"{src_lbl}; logs/S3_k3_retrieved*/{RESULT}")


def evalnoise_stats():
    pairs = [("S1r_k1_retrieved", "S1r_k1_retrieved_ns1"), ("S1_k1_random", "S1_k1_random_ns1"),
             ("S2_k3_random", "S2_k3_random_ns3"), ("S3_k3_retrieved", "S3_k3_retrieved_ns3")]
    d_mean, d_cls = [], []
    for run, grid in pairs:
        a = parse_result(f"{REPO}/logs/{run}/{RESULT}")
        b = parse_result(f"{REPO}/results/grid/{grid}/{RESULT}")
        d_mean.append(abs(a["mIoU"] - b["mIoU"]))
        d_cls += [abs(a[c] - b[c]) for c in a if c != "mIoU"]
    src = "logs/<run>/" + RESULT + " vs results/grid/<run>_ns<k>/" + RESULT
    add("evalnoise", "same checkpoint scored twice: max |delta| mIoU / max |delta| any class",
        f"{max(d_mean):.2f} / {max(d_cls):.2f}",
        f"{len(pairs)} checkpoints (S1, S1r, S2, S3), scored by the ablation and again by the "
        "num_src grid at the training k; results are printed to 0.01, so 0.00 means < 0.005", src)
    for arm, runs in SEEDS.items():
        v = [parse_result(f"{REPO}/logs/{r}/{RESULT}")["mIoU"] for r in runs]
        add("evalnoise", f"{arm} training seeds: mIoU values / range / sd",
            f"{' / '.join(f'{x:.2f}' for x in v)} / {max(v) - min(v):.2f} / {np.std(v, ddof=1):.2f}",
            "three trainings of one config; S0's third is ratio_smol_p0.5, the same config",
            f"logs/{{{','.join(runs)}}}/{RESULT}")
    s3 = [parse_result(f"{REPO}/logs/{r}/{RESULT}")["mIoU"] for r in SEEDS["S3"]]
    s0 = [parse_result(f"{REPO}/logs/{r}/{RESULT}")["mIoU"] for r in SEEDS["S0"]]
    train_sd = float(np.mean([np.std(s3, ddof=1), np.std(s0, ddof=1)]))
    floor = max(max(d_mean), 0.005)
    add("evalnoise", "training sd / evaluation |delta| (mIoU)", f">= {train_sd / floor:.0f}x",
        "mean of S0 and S3 seed sd over the largest re-scoring difference, which is floored at "
        "0.005, half the printed precision, so the ratio is a lower bound", "rows above")


def main():
    subs = subsets()
    corpus_stats(subs)
    coreg_stats(subs)
    thin_stats(subs)
    evalnoise_stats()
    out = f"{REPO}/results/paper_stats.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "statistic", "value", "definition", "inputs", "script"])
        w.writerows(rows)
    for g, s, v, *_ in rows:
        print(f"{g:<9} {s:<90} {v}")
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
