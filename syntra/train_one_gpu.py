# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import os
import sys
import logging
import random
import argparse
from datetime import datetime
import shutil
from os.path import join
from tqdm import tqdm

    
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_module


from syntra.modeling.dataset import get_dataset
from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling.losses import get_loss_fn
from syntra.utils.vis import vis_train
from syntra.modeling.validator import Validator
from syntra.modeling import build_segmentation_model
from syntra.utils.train_utils import adjust_learning_rate_cosine, adjust_learning_rate_exp_decay
from syntra.utils.sampler import get_sampler
from syntra.utils.debug_utils import *
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict, OmegaConf

# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")


# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def set_seeds(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path



def parse_args():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    # common parameters
    parser.add_argument("-t", "--task-name", type=str, help="task name")
    parser.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    # model
    parser.add_argument("-r", "--dora-rank", type=int, help="dora rank")
    parser.add_argument("-mn", "--model-name", type=str, help="model name: smol or syntra")
    parser.add_argument("--backbone", type=str, help="backbone type: vit_h, vit_l, vit_b, hiera_t, hiera_s, hiera_l, hiera_bp")
    parser.add_argument("--prompt-encoder", type=str, help="prompt encoder type: example, notion")
    parser.add_argument("--mask-decoder", type=str, help="mask decoder type: default, highres")
    # dataset
    parser.add_argument("-d", "--root-dir", type=str, help="path to training datasets")
    parser.add_argument("--dataset-name", type=str, help="dataset name: smol or syntra")
    parser.add_argument("--data-mode", type=str, help="sample list file (.txt).")
    parser.add_argument("--data-info", type=str, help="data info file (.json)")
    parser.add_argument("--img-size", type=int, help="input image size")
    parser.add_argument("--num-src", type=int, help="number of source images")
    parser.add_argument("--num-notions", type=int, help="number of notions")
    parser.add_argument("-fg", "--foreground-ratio", type=float, help="foreground ratio in a training patch")
    parser.add_argument("-val", "--validation-data", type=str, help="path to validation dataset")
    parser.add_argument("--sampler", type=str, help="sampler name: random, weighted_random, sequential")
    parser.add_argument("-dkw", "--data-keywords", type=str, help='dataset key words for filtering, e.g., railway, vineyard, bayern, hameln')
    # train
    parser.add_argument("-e", "--num-epochs", type=int, help="number of training epochs")
    parser.add_argument("-itr", "--max-iterations", type=int, help="maximum training iterations")
    parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
    parser.add_argument("-nw", "--num-workers", type=int, help="number of data loading workers")
    parser.add_argument("-a", "--accum-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--cache-data", action="store_true", help="cache dataset into memory")
    # Optimizer parameters
    parser.add_argument("--weight-decay", type=float, help="weight decay for optimizer")
    parser.add_argument("--lr", type=float, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--warmup-itrs", type=int, metavar="warmup epochs", help="total number of warmup epochs")
    parser.add_argument("--use-amp", type=bool, help="use amp")
    parser.add_argument("--resume", type=str, help="Resuming training from checkpoint")
    parser.add_argument("--vis-freq", type=int, help="visualization frequency (in iterations)")
    # parser.add_argument("--train-patch-emb", action="store_true", default=False, help="Train patch embedding parameters")
    parser.add_argument("--lr-scheduler", type=str, help="learning rate scheduler: cosine, exp or none")
    parser.add_argument("--from-scratch", type=bool, help="train from scratch")
    parser.add_argument("--disable-runid", type=bool, help="disable run id in output folder name")
    args = parser.parse_args()

    # print all available hydra config files
    if not GlobalHydra().is_initialized():
        initialize_config_module(config_module="syntra.configs", version_base=None)
    
    hydra_cfg = compose(config_name=args.config)
    cfg = hydra_cfg['TRAIN']

    # Override cfg arguments with args if provided
    with open_dict(cfg):
        for key, value in vars(args).items():
            if value is not None:
                setattr(cfg, key, value)

    # save the final haydra config to yaml file
    with open_dict(hydra_cfg):
        hydra_cfg["TRAIN"] = cfg
    cfg_yaml_str = OmegaConf.to_yaml(hydra_cfg)
    with open(join(cfg.output_dir, os.path.basename(args.config).split(".")[0], "config.yaml"), "w") as f:
        f.write(cfg_yaml_str)
    
    return cfg



def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    set_seeds(1234)
    if args.task_name is None:
        args.task_name = os.path.splitext(args.config)[0]
    if args.disable_runid:
        model_save_path = join(args.output_dir, args.task_name)
    else:
        model_save_path = join(args.output_dir, args.task_name + "-" + run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # %% set up model
    os.makedirs(model_save_path, exist_ok=True)

    # save cfg to yaml file
    yaml_str = OmegaConf.to_yaml(args)
    with open(join(model_save_path, "config.yaml"), "w") as f:
        f.write(yaml_str)

    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    logger = logging.getLogger("SYNTRA")
    logging.basicConfig(filename=join(model_save_path, "log.txt"), level=logging.INFO)

    model = build_segmentation_model(**args).to(device)
    model.train()

    logger.info(">>>>>>>>>>>>>>>>>>CONFIGS<<<<<<<<<<<<<<<<")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"# total parameters: "
                f"{sum(p.numel() for p in model.parameters()) / 1024**2:.2f} M")  
    logger.info(f"# trainable parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024**2:.2f} M")  
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_fn = get_loss_fn("SAMLoss")

    # dataset
    uni_transform = TransformCompose(args.uni_transform) if getattr(args, 'uni_transform', None) is not None else None
    aug_transform = TransformCompose(args.aug_transform) if getattr(args, 'aug_transform', None) is not None else None
    kwargs = {}
    if hasattr(args, 'pos_prob'):
        kwargs['pos_prob'] = args.pos_prob
    train_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               num_src=args.num_src,
                               num_notions = args.num_notions,
                               mode="train",
                               info_file=args.data_info,
                               fg_ratio=args.foreground_ratio,
                               filtering_keys=args.data_keywords,
                               img_size=args.img_size,
                               cache_data=args.cache_data,
                               uni_transform=uni_transform,
                               aug_transform=aug_transform,
                               **kwargs)
    logger.info(f"# training samples: {len(train_dataset)}")
    sampler, batch_sampler = get_sampler(train_dataset, batch_size=args.batch_size, sampler_name=args.sampler) 
    if batch_sampler is None:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            sampler=sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,         # avoids respawning workers each epoch
            prefetch_factor=2,               # default is 2; lower if you get shm issues
            collate_fn=train_dataset.collate_fn,
        )

    # validation dataset
    validator = None
    if args.validation_data is not None:
        val_dataset = get_dataset(args.dataset_name,
                                 data_root=args.root_dir,
                                 mode="train",
                                 info_file=args.data_info,
                                 fg_ratio=args.foreground_ratio)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )
        validator = Validator(val_dataloader, device)

    # %% train
    num_itr_per_epoch = len(train_dataloader)
    num_epochs = max(args.num_epochs, args.max_iterations // num_itr_per_epoch)
    max_iterations = max(args.max_iterations, num_epochs * num_itr_per_epoch)
    warmup_epochs = round(args.warmup_itrs / num_itr_per_epoch)
    logger.info(f"# epochs: {num_epochs}")
    iter_num = 1
    losses = []
    lrs = []
    best_loss = 1e10
    max_mean_iou = 0
    start_epoch = 1

    if getattr(args, 'resume', None) is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_trainable_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            iter_num = len(train_dataloader) * checkpoint["epoch"] + 1
    if getattr(args, 'use_amp', False):
        scaler = torch.cuda.amp.GradScaler()

    logger.info(">>>>>>>>>>>>>>>>>>START TRAINING <<<<<<<<<<<<<<<<<<")
    pbar = tqdm(range(max_iterations))
    pbar.update(iter_num - 1)
    # pbar.set_postfix({'Epoch/Step': f'{epoch}/0', 'ITR': iter_num, 'LR': f'{cur_lr:e}', 'Loss': f'{0:.6f}'})
    for epoch in range(start_epoch, num_epochs + 1):
        inter_avg_loss = 0
        if args.lr_scheduler == 'cosine':
            cur_lr = adjust_learning_rate_cosine(optimizer, base_lr=args.lr, epoch=epoch,
                                                 total_epoch=args.num_epochs,
                                                 warmup_epochs=warmup_epochs)
        else:
            cur_lr = optimizer.param_groups[0]["lr"]

        for step, data in enumerate(train_dataloader):
            if args.lr_scheduler == 'exp':
                cur_lr = adjust_learning_rate_exp_decay(optimizer, base_lr=args.lr, itr=iter_num,
                                                        max_itr=max_iterations,
                                                        warmup_itrs=args.warmup_itrs)
            # optimizer.zero_grad()
            img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
            aux = data[4] if len(data) > 4 else None
            # Prepare training data on GPU if needed
            if hasattr(train_dataset, 'prepare_inputs'):
                img_src, lbl_src, img_tgt, lbl_tgt, aux = train_dataset.prepare_inputs(
                    img_src, lbl_src, img_tgt, lbl_tgt, aux)
            if getattr(args, 'use_amp', False):
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(img_src, lbl_src, img_tgt, aux)
                    loss = loss_fn(pred, lbl_tgt **{'aux': aux}) / args.accum_steps
                    # loss = ce_loss(mapsam_pred, lbl_tgt.float())
                scaler.scale(loss).backward()
                if iter_num % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred = model(img_src, lbl_src, img_tgt, aux)
                loss = loss_fn(pred, lbl_tgt, **{'aux': aux}) / args.accum_steps
                # loss = ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1))
                loss.backward()
                if iter_num % args.accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            inter_avg_loss += loss.item()
            iter_num += 1
            if iter_num % 10 == 0:
                pbar.update(10)
                pbar.set_postfix({'ITR': iter_num, 'LR': f'{cur_lr:e}', 'Loss': f'{inter_avg_loss / (step + 1):.6f}'})
                logger.info({'ITR': iter_num, 'LR': f'{cur_lr:e}', 'Loss': f'{inter_avg_loss / (step + 1):.6f}'})
            

            if args.vis_freq is not None and iter_num % args.vis_freq == 0:
                vis_train(img_src, lbl_src, img_tgt, lbl_tgt, pred, aux, iter_num)

            if iter_num % 100 == 0:
                inter_avg_loss /= max(step, 1)
                losses.append(inter_avg_loss)
                lrs.append(cur_lr)
                logger.info(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {inter_avg_loss}')
                # get saving parameters
                state_dict = model.trainable_state_dict()
                ## save the latest model
                checkpoint = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, join(model_save_path, "model_latest.pth"))
                ## save the best model
                save_best_flag = False
                if validator is not None:
                    if epoch % (2000 // num_itr_per_epoch) == 0:
                        print("Validating...")
                        val_mean_iou, val_res, val_str = validator.validate(model)
                        logger.info('\n' + val_str)
                        print(val_str)
                        if val_mean_iou > max_mean_iou:
                            save_best_flag = True
                            max_mean_iou = val_mean_iou
                elif inter_avg_loss < best_loss:
                    save_best_flag = True
                    best_loss = inter_avg_loss

                if save_best_flag:
                    checkpoint = {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, join(model_save_path, "model_best.pth"))

                ## save the intermediate model
                if iter_num % 1e4 == 0:
                    checkpoint = {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, join(model_save_path, f"model_{iter_num}.pth"))

                # save loss
                with open(join(model_save_path, "loss.txt"), "w") as f:
                    for x, y in zip(losses, lrs):
                        f.write(f'{y:.6f}, {x:.6f}\n')
                

if __name__ == "__main__":
    main()
