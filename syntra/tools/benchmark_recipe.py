# -*- coding: utf-8 -*-
"""Measure training throughput and memory for batch size and mixed precision.

Reports samples per second and peak memory for each (batch size, AMP) combination, and
projects the wall-clock of one full run. The projection is made at constant *work*: the
reference recipe is 100k iterations of batch 8 with 4 accumulation steps, i.e. 800k samples at
an effective batch of 32. A larger batch therefore has to run proportionally fewer iterations
with proportionally fewer accumulation steps to train on the same data.

Note that each sample carries `num_src` source tiles plus one target, so the image encoder sees
batch x (num_src + 1) images per iteration - that, not the batch size itself, drives memory.

    PYTHONPATH=. python syntra/tools/benchmark_recipe.py \\
        --config syntra/syntra_maps_s100.yaml --batch-sizes 8,12,16 --iters 30
"""

import argparse
import time

import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader

from syntra.modeling import build_segmentation_model
from syntra.modeling.dataset import get_dataset
from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling.losses import get_loss_fn
from syntra.utils.sampler import get_sampler

REFERENCE_SAMPLES = 100_000 * 8       # the current recipe: 100k iterations at batch 8
REFERENCE_EFFECTIVE_BATCH = 32        # batch 8 x 4 accumulation steps


def build(cfg, args, batch_size):
    dataset = get_dataset(
        cfg.dataset_name, data_root=args.root_dir or cfg.root_dir, info_file=cfg.data_info,
        num_src=cfg.num_src, num_notions=cfg.num_notions, mode="train",
        fg_ratio=cfg.foreground_ratio, filtering_keys=args.data_keywords or cfg.data_keywords,
        exclude_keys=cfg.get("exclude_keys", None), img_size=cfg.img_size,
        uni_transform=TransformCompose(cfg.uni_transform),
        aug_transform=TransformCompose(cfg.aug_transform),
        src_selection=cfg.get("src_selection", "retrieved"),
    )
    sampler, batch_sampler = get_sampler(dataset, batch_size=batch_size, sampler_name=cfg.sampler)
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=dataset.collate_fn)
    return dataset, loader


def run(cfg, args, batch_size, use_amp, device):
    dataset, loader = build(cfg, args, batch_size)
    model = build_segmentation_model(
        model_name=cfg.model_name, backbone=cfg.backbone, dora_rank=cfg.dora_rank,
        prompt_encoder=cfg.prompt_encoder, mask_decoder=cfg.mask_decoder,
        src_fusion=cfg.get("src_fusion", "max"),
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = get_loss_fn("SAMLoss", ce_neg_weight=cfg.ce_neg_weight).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    torch.cuda.reset_peak_memory_stats()
    seen, started, losses = 0, None, []
    iterator = iter(loader)
    for step in range(args.warmup + args.iters):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            data = next(iterator)

        img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device, non_blocking=True) for d in data[:4]]
        aux = data[4]
        img_src, lbl_src, img_tgt, lbl_tgt, aux = dataset.prepare_inputs(
            img_src, lbl_src, img_tgt, lbl_tgt, aux)

        # mirror the real loop: gradients accumulate and the optimizer steps every
        # accum_steps iterations, so the effective batch stays at batch_size x accum_steps
        accum = max(1, int(round(REFERENCE_EFFECTIVE_BATCH / batch_size)))
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(img_src, lbl_src, img_tgt)
                loss, _ = loss_fn(pred, lbl_tgt, aux=aux)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            pred = model(img_src, lbl_src, img_tgt)
            loss, _ = loss_fn(pred, lbl_tgt, aux=aux)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step == args.warmup - 1:          # start timing after warm-up
            torch.cuda.synchronize()
            started = time.perf_counter()
        elif step >= args.warmup:
            seen += img_tgt.shape[0]
            losses.append(float(loss))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return seen / elapsed, peak, sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("-d", "--root-dir", default=None)
    parser.add_argument("-dkw", "--data-keywords", default=None)
    parser.add_argument("--batch-sizes", default="8,12,16")
    parser.add_argument("--amp", default="off,on")
    parser.add_argument("--iters", type=int, default=30, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("-nw", "--num-workers", type=int, default=8)
    args = parser.parse_args()

    GlobalHydra.instance().clear()
    module, name = args.config.split("/") if "/" in args.config else ("", args.config)
    initialize_config_module(config_module=f"syntra.configs.{module}".rstrip("."), version_base=None)
    cfg = compose(config_name=name)["TRAIN"]
    device = torch.device("cuda")

    print(f"{torch.cuda.get_device_name(0)}, num_src={cfg.num_src} "
          f"(encoder sees batch x {cfg.num_src + 1} images), img_size={cfg.img_size}\n")
    header = f"{'batch':>6} {'amp':>4} {'samples/s':>10} {'peak GB':>8} {'h / full run':>13} {'mean loss':>10}"
    print(header)
    print("-" * len(header))

    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        for amp in [a.strip() == "on" for a in args.amp.split(",")]:
            try:
                rate, peak, loss = run(cfg, args, batch_size, amp, device)
                hours = REFERENCE_SAMPLES / rate / 3600
                print(f"{batch_size:>6} {'on' if amp else 'off':>4} {rate:>10.2f} {peak:>8.1f} "
                      f"{hours:>13.1f} {loss:>10.4f}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{batch_size:>6} {'on' if amp else 'off':>4} {'OOM':>10}")

    print("\nProjection is at constant work: 800k samples at an effective batch of 32, so a "
          "larger batch needs proportionally fewer iterations and accumulation steps.")


if __name__ == "__main__":
    main()
