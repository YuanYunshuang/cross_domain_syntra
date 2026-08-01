# -*- coding: utf-8 -*-
"""Sweep the source-selection strategy and the number of sources at inference time (E1).

Evaluates one trained checkpoint under several (src_selection, num_src, src_fusion)
combinations. All combinations see the **same** target tiles - only the source examples that
prompt the model change - so the comparison is paired and the difference is attributable to
the retrieval mechanism alone.

    PYTHONPATH=. python syntra/tools/retrieval_sweep.py \\
        --config syntra/syntra_maps_s100.yaml \\
        -w /koko/SYNTRA_logs/syntra/syntra_hismap_n1_s100_v3/model_latest.pth \\
        -dkw hismap -n 800 -o results/e1_inference_sweep.csv
"""

import argparse
import csv
import os

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from syntra.modeling import build_segmentation_model
from syntra.modeling.dataset import get_dataset
from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.tools.eval import Evaluator


def load_config(config_name):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    if len(config_name.split("/")) == 2:
        module, config_name = config_name.split("/")
        module = "syntra.configs." + module
    else:
        module = "syntra.configs"
    initialize_config_module(config_module=module, version_base=None)
    return compose(config_name=config_name)


def build_dataset(cfg, args, src_selection, num_src):
    return get_dataset(
        cfg.dataset_name,
        data_root=args.root_dir or cfg.root_dir,
        info_file=args.data_info or cfg.data_info,
        num_notions=cfg.num_notions,
        num_src=num_src,
        fg_ratio=cfg.foreground_ratio,
        filtering_keys=args.data_keywords or cfg.data_keywords,
        mode="test",
        img_size=cfg.img_size,
        uni_transform=TransformCompose(cfg.uni_transform),
        src_selection=src_selection,
        exclude_keys=args.exclude_keys if getattr(args, "exclude_keys", None) is not None
        else cfg.get("exclude_keys", None),
    )


def stratified_indices(dataset, per_class, seed, negatives_per_class):
    """Pick target tiles per (class, subdataset) group, positives and negatives separately.

    ``file_list`` entries look like ``forest.hameln.pos/hameln.3922_1898.0_0``. Positives and
    negatives are drawn in separate quotas on purpose: IoU is accumulated globally per class,
    so negatives only enter through their false positives, but a subset whose mix of the two
    drifts between configurations would not be comparable. Fixing both quotas keeps the
    comparison paired.

    Negative targets - the queried class is absent - are what measure false-positive
    suppression, which matters here because ``max`` fusion takes the maximum over the sources:
    every additional source is another chance to fire on a target that has no foreground.
    """
    pos_groups, neg_groups = {}, {}
    for i, entry in enumerate(dataset.file_list):
        prefix = entry.split("/")[0]
        group = prefix.rsplit(".", 1)[0]
        (neg_groups if prefix.endswith(".neg") else pos_groups).setdefault(group, []).append(i)

    rng = np.random.RandomState(seed)
    indices = []
    for group in sorted(set(pos_groups) | set(neg_groups)):
        pos_pool, neg_pool = pos_groups.get(group, []), neg_groups.get(group, [])
        n_pos = min(per_class, len(pos_pool))
        if negatives_per_class < 0:
            # natural prevalence: keep the class's true positive/negative ratio, which is what
            # a sliding window over a whole map sheet actually meets
            ratio = len(neg_pool) / max(len(pos_pool), 1)
            n_neg = min(int(round(n_pos * ratio)), len(neg_pool))
        else:
            n_neg = min(negatives_per_class, len(neg_pool))
        if n_pos:
            indices.extend(rng.choice(pos_pool, n_pos, replace=False).tolist())
        if n_neg:
            indices.extend(rng.choice(neg_pool, n_neg, replace=False).tolist())
        print(f"    {group:<28} {n_pos:>5} pos of {len(pos_pool):<6} {n_neg:>5} neg of {len(neg_pool)}")
    return sorted(indices)


@torch.no_grad()
def evaluate(model, dataset, indices, batch_size, num_workers, device, seed):
    """Evaluate on a fixed set of target indices. Returns (per-class dict, mean IoU)."""
    # the source draw is random for "retrieved"/"random"; seed it so runs are reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate_fn)
    evaluator = Evaluator()
    n_neg = 0
    neg_fg_fraction = 0.0   # mean share of pixels wrongly predicted as foreground
    n_false_alarms = 0      # negatives with more than 1% of pixels predicted as foreground

    for data in tqdm(loader, leave=False):
        img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
        meta = data[4]
        if hasattr(dataset, "prepare_inputs"):
            img_src, lbl_src, img_tgt, lbl_tgt, meta = dataset.prepare_inputs(
                img_src, lbl_src, img_tgt, lbl_tgt, meta)
        pred = model(img_src, lbl_src, img_tgt).sigmoid()
        evaluator.add_sample(pred.squeeze(2), lbl_tgt, meta=meta)

        # negatives: no ground-truth foreground for the queried class, so IoU is undefined per
        # sample and only the predicted foreground says anything
        gt = lbl_tgt.flatten(2)
        is_neg = (gt > 0).sum(-1) == 0                              # (B, N)
        if is_neg.any():
            fg = (pred.squeeze(2) > 0.5).flatten(2).float().mean(-1)  # (B, N)
            neg_fg = fg[is_neg]
            n_neg += int(is_neg.sum())
            neg_fg_fraction += float(neg_fg.sum())
            n_false_alarms += int((neg_fg > 0.01).sum())

    per_class = {k: float(v[0]) * 100 for k, v in evaluator.eval().items()}
    mean = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    negatives = {
        "n_neg": n_neg,
        "neg_fg_pct": round(100 * neg_fg_fraction / n_neg, 3) if n_neg else float("nan"),
        "false_alarm_pct": round(100 * n_false_alarms / n_neg, 2) if n_neg else float("nan"),
    }
    return per_class, mean, negatives


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config name, e.g. syntra/syntra_maps_s100.yaml")
    parser.add_argument("-w", "--checkpoint", required=True)
    parser.add_argument("-d", "--root-dir", default=None)
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("-xkw", "--exclude-keys", default=None)
    parser.add_argument("--data-info", default=None)
    parser.add_argument("-n", "--num-samples", type=int, default=150,
                        help="target tiles per class; the same ones for every combination")
    parser.add_argument("--negatives", type=int, default=None,
                        help="negative target tiles per class (default: same as --num-samples). "
                             "Negatives carry no ground-truth foreground, so they enter the IoU "
                             "only through their false positives and are additionally reported "
                             "as neg_fg_pct and false_alarm_pct. Set 0 to evaluate positives "
                             "only, or -1 to keep each class's natural positive/negative ratio, "
                             "which is what a sliding window over a whole map sheet meets.")
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-nw", "--num-workers", type=int, default=8)
    parser.add_argument("--selections", default="retrieved,topk,random")
    parser.add_argument("--ks", default="1,3,7", help="numbers of source examples to try")
    parser.add_argument("--fusions", default="max")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-o", "--output", default="results/e1_inference_sweep.csv")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    cfg = full_cfg["TRAIN"]
    # evaluation uses the TEST block of the config unless overridden on the command line
    test_cfg = full_cfg.get("TEST", {}) or {}
    if args.data_info is None:
        args.data_info = test_cfg.get("data_info", cfg.data_info)
    if args.data_keywords is None:
        args.data_keywords = test_cfg.get("data_keywords", cfg.data_keywords)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selections = [s.strip() for s in args.selections.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    fusions = [f.strip() for f in args.fusions.split(",")]

    # Fixed target subset, identical for every combination and stratified per class, so that
    # every class contributes comparable evidence regardless of how rare it is.
    probe = build_dataset(cfg, args, selections[0], ks[0])
    n_neg_per_class = args.num_samples if args.negatives is None else args.negatives
    indices = stratified_indices(probe, args.num_samples, args.seed, n_neg_per_class)
    print(f"Evaluating {len(indices)} of {len(probe)} test targets "
          f"(up to {args.num_samples} positives and {n_neg_per_class} negatives per class), "
          f"identical across all combinations")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = []
    for fusion in fusions:
        model = build_segmentation_model(
            model_name=cfg.model_name, backbone=cfg.backbone, dora_rank=cfg.dora_rank,
            prompt_encoder=cfg.prompt_encoder, mask_decoder=cfg.mask_decoder, src_fusion=fusion,
        )
        model.load_trainable_state_dict(state["model"])
        model.to(device).eval()

        for selection in selections:
            for k in ks:
                dataset = build_dataset(cfg, args, selection, k)
                per_class, mean, negatives = evaluate(model, dataset, indices, args.batch_size,
                                                      args.num_workers, device, args.seed)
                print(f"  {selection:<10} k={k}  fusion={fusion:<4}  mIoU={mean:6.2f}  "
                      f"neg_fg={negatives['neg_fg_pct']:.3f}%  "
                      f"false_alarms={negatives['false_alarm_pct']:.1f}%")
                rows.append({"selection": selection, "num_src": k, "fusion": fusion,
                             "mIoU": round(mean, 2), "n_targets": len(indices), **negatives,
                             **{c: round(v, 2) for c, v in per_class.items()}})
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    keys = ["selection", "num_src", "fusion", "mIoU", "neg_fg_pct", "false_alarm_pct",
            "n_targets", "n_neg"]
    keys += sorted({k for r in rows for k in r if k not in keys})
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
