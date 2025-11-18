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
from syntra.modeling.losses import get_loss_fn
from syntra.utils.vis import vis_train
from syntra.modeling.validator import Validator
from syntra.modeling import build_segmentation_model
from syntra.utils.train_utils import adjust_learning_rate_cosine, adjust_learning_rate_exp_decay
from syntra.utils.sampler import get_sampler
from syntra.utils.debug_utils import *
from hydra.core.global_hydra import GlobalHydra

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
    parser.add_argument("-t", "--task-name", type=str, default=None, help="task name")
    parser.add_argument("-w", "--checkpoint", type=str, default=f"{cur_path}/../model_zoo/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("-o", "--output-dir", type=str, default=f"{cur_path}/logs")
    # model
    parser.add_argument("-r", "--dora-rank", type=int, default=16)
    parser.add_argument("-mn", "--model-name", type=str, default="smol", help="model name: unet or syntra")
    parser.add_argument("--backbone", type=str, default="vit_b", help="backbone type: vit_h, vit_l, vit_b, hiera_t, hiera_s, hiera_l, hiera_bp")
    parser.add_argument("--prompt-encoder", type=str, default="example", help="prompt encoder type: example, notion")
    parser.add_argument("--mask-decoder", type=str, default="default", help="mask decoder type: default, highres")
    # dataset
    parser.add_argument("-d", "--root-dir", type=str, default=None, help="path to training datasets")
    parser.add_argument("--dataset-name", type=str, default="syntra", help="dataset name: smol or syntra")
    parser.add_argument("--data-mode", type=str, default="nshot100", help="sample list file (.txt).")
    parser.add_argument("--data-info", type=str, help="data info file (.json)")
    parser.add_argument("--img-size", type=int, default=384, help="input image size")
    parser.add_argument("--num-src", type=int, default=1, help="number of source images")
    parser.add_argument("--num-notions", type=int, default=1, help="number of notions")
    parser.add_argument("-fg", "--foreground-ratio", type=float, default=0.05)
    parser.add_argument("-val", "--validation-data", type=str, default=None)
    parser.add_argument("--sampler", type=str, default="uniform", help="sampler type: uniform or prob")
    parser.add_argument("-dkw", "--data-keywords", type=str, default=None, help='dataset key words for filtering, e.g., railway, vineyard, bayern, hameln')
    # train
    parser.add_argument("-e", "--num-epochs", type=int, default=200)
    parser.add_argument("-itr", "--max-iterations", type=int, default=100000)
    parser.add_argument("-bs", "--batch-size", type=int, default=16)
    parser.add_argument("-nw", "--num-workers", type=int, default=4)
    parser.add_argument("-a", "--accum-steps", type=int, default=4)
    parser.add_argument("--freeze-prompt-encoder", action="store_true", default=False, help="Train prompt encoder parameters")
    # Optimizer parameters
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--warmup-itrs", type=int, default=1000, metavar="warmup epochs", help="total number of warmup epochs")
    parser.add_argument("--use-wandb", type=bool, default=False, help="use wandb to monitor training")
    parser.add_argument("--use-amp", action="store_true", default=False, help="use amp")
    parser.add_argument("--resume", type=str, default=None, help="Resuming training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plot-loss", action="store_true", default=False)
    parser.add_argument("--vis-freq", type=int, default=None, help="visualization frequency (in iterations)")
    # parser.add_argument("--train-patch-emb", action="store_true", default=False, help="Train patch embedding parameters")
    parser.add_argument("--lr-scheduler", type=str, default="exp", help="learning rate scheduler: cosine, exp or none")
    parser.add_argument("--from-scratch", action="store_true", default=False)
    parser.add_argument("--disable-runid", action="store_true", default=False)
    parser.add_argument("--save-dora-only", type=bool, default=True)
    args = parser.parse_args()

    # print all available hydra config files
    if not GlobalHydra().is_initialized():
        initialize_config_module(config_module="syntra.configs", version_base=None)
    
    cfg = compose(config_name=args.config)['TRAIN']

    # Override args with cfg arguments
    for key, value in cfg.items():
        # args that have higher priority in cmd args than cfg to facilitate quick root dir switching
        arg_list = ['root_dir', 'output_dir', 'resume']
        if key in arg_list and getattr(args, key) is not None:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args



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
    device = torch.device(args.device)
    # %% set up model
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    logger = logging.getLogger("SYNTRA")
    logging.basicConfig(filename=join(model_save_path, "log.txt"), level=logging.INFO)

    if args.from_scratch:
        args.checkpoint = None
        args.warmup_epochs = 1

    model = build_segmentation_model(**vars(args)).to(device)
    model.train()

    logger.info(">>>>>>>>>>>>>>>>>>CONFIGS<<<<<<<<<<<<<<<<")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"# total parameters: "
                f"{sum(p.numel() for p in model.parameters()) / 1024**2:.2f} M")  
    logger.info(f"# trainable parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024**2:.2f} M")  
    if args.freeze_prompt_encoder:
        opt_params = list(model.image_encoder.parameters()) + list(
            model.mask_decoder.parameters()
        )
    else:
        opt_params = model.parameters()
    optimizer = torch.optim.AdamW(
        opt_params, lr=args.lr, weight_decay=args.weight_decay
    )

    loss_fn = get_loss_fn("SAMLoss")

    # dataset
    train_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               num_src=args.num_src,
                               num_notions = args.num_notions,
                               mode="train",
                               info_file=args.data_info,
                               fg_ratio=args.foreground_ratio,
                               filtering_keys=args.data_keywords,
                               img_size=args.img_size)
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
            # persistent_workers=True,         # avoids respawning workers each epoch
            # prefetch_factor=2,               # default is 2; lower if you get shm issues
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

    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_trainable_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            iter_num = len(train_dataloader) * checkpoint["epoch"] + 1
    if args.use_amp:
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
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mapsam_pred = model(img_src, lbl_src, img_tgt)
                    loss = loss_fn(mapsam_pred, lbl_tgt **{'aux': aux}) / args.accum_steps
                    # loss = ce_loss(mapsam_pred, lbl_tgt.float())
                scaler.scale(loss).backward()
                if iter_num % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                mapsam_pred = model(img_src, lbl_src, img_tgt)
                loss = loss_fn(mapsam_pred, lbl_tgt, **{'aux': aux}) / args.accum_steps
                # loss = ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1))
                loss.backward()
                if iter_num % args.accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            inter_avg_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'Epoch/Step': f'{epoch}/{step + 1}', 'ITR': iter_num, 'LR': f'{cur_lr:e}', 'Loss': f'{inter_avg_loss / (step + 1):.6f}'})
            iter_num += 1
            if iter_num % 10 == 0:
                logger.info({'Epoch/Step': f'{epoch}/{step + 1}', 'ITR': iter_num, 'LR': f'{cur_lr:e}', 'Loss': f'{inter_avg_loss / (step + 1):.6f}'})
            

            if args.vis_freq is not None and iter_num % args.vis_freq == 0:
                vis_train(img_src, lbl_src, img_tgt, lbl_tgt, mapsam_pred, aux, iter_num)

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
