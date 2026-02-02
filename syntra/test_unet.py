# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import os, sys
import logging

join = os.path.join
from omegaconf import open_dict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
import argparse


from syntra.modeling.dataset import get_dataset
from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling import build_segmentation_model
from syntra.tools.eval import UnetEvaluator as Evaluator
from syntra.utils.debug_utils import *
from hydra import compose, initialize_config_module
from syntra.modeling.losses import get_loss_fn

# set seeds
torch.manual_seed(2025)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6



def setup_parser():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("-w", "--checkpoint", type=str, help="path to model checkpoint")
    parser.add_argument("-o","--output_dir", type=str)
    parser.add_argument("-nw", "--num_workers", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-s", "--save-result", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)
    # dataset
    parser.add_argument("-d", "--root-dir", type=str, default=None, help="path to training dataset; two subfolders: lbls and imgs")
    parser.add_argument("-dkw", "--data-keywords", type=str, default=None, help='dataset key words for filtering')
    parser.add_argument("--dataset-name", type=str, )
    parser.add_argument("--data-info", type=str, )
    parser.add_argument("--img-size", type=int)
    args = parser.parse_args()

    initialize_config_module(config_module="syntra.configs", version_base=None)
    cfg = compose(config_name=args.config)
    cfg_test = cfg['TRAIN'].copy() # initialize with TRAIN config

    # 1. Override cfg['TRAIN'] parameters with TEST parameters if not None
    # 2. Override cfg['TRAIN'] parameters with args parameters if not None
    with open_dict(cfg_test):
        for key, value in cfg['TEST'].items():
            if value is not None or not hasattr(cfg_test, key):
                setattr(cfg_test, key, value)
        for key, value in vars(args).items():
            if value is not None:
                setattr(cfg_test, key, value)

    print(cfg_test)

    return cfg_test


def single_test(args, device, model, seg_loss, logger, out_dir):
    losses = []
    uni_transform = TransformCompose(args.uni_transform) if getattr(args, 'uni_transform', None) is not None else None
    test_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               info_file=args.data_info,
                               filtering_keys=args.data_keywords,
                               mode="test",
                               img_size=args.img_size,
                               uni_transform=uni_transform,)
    num_samples = len(test_dataset)

    logger.info(f"# testing samples: {num_samples}")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )
    id_to_name = {i+1: k for i, k in enumerate(test_dataset.color_map.keys())}

    evaluator = Evaluator()
    with (torch.no_grad()):
        pbar = tqdm(test_dataloader)
        for step, data in enumerate(pbar):
            imgs, lbls = [d.to(device) for d in data[:2]]
            subsets = data[2]
            if hasattr(test_dataset, 'prepare_inputs'):
                imgs, lbls = test_dataset.prepare_inputs(imgs, lbls)

            model_pred = model(imgs)
            lbl_pred = model_pred.argmax(dim=1)
            loss = seg_loss(model_pred, lbls)[0].item()
            
            pbar.set_postfix({ 'Loss': f'{loss:.6f}'})
            losses.append(loss)
            evaluator.add_sample(lbl_pred, lbls.squeeze(1), subsets, id_to_name)

    avg_loss = np.array(losses).mean()
    eval_out = []
    out_info = f"Average Loss: {avg_loss:.6f}\n" + "=" * 20 +\
        " Evaluation Results " + "=" * 20 + "\n"
    
    avg_eval_dict = evaluator.eval()
    for notion_name,  avg_eval in avg_eval_dict.items():
        eval_out.append(avg_eval)
        out_info += \
            f"Notion set : {notion_name}: \n" + \
            f"  IoU (FG): {avg_eval[0]*100:.2f}\n" + \
            f"  Precision (FG): {avg_eval[1]*100:.2f}\n" + \
            f"  Recall (FG): {avg_eval[2]*100:.2f}\n" + \
            f"  Accuracy (BG&FG): {avg_eval[3]*100:.2f}\n" + \
            "-" * 40 + "\n"
    logger.info(out_info)
    print(out_info)
    return eval_out


def main():
    args = setup_parser()
    if args.checkpoint is None:
        assert args.output_dir is not None, "Either checkpoint or output_dir must be provided."
        out_dir = os.path.join(args.output_dir, args.config.split('.')[0])
        args.checkpoint = os.path.join(out_dir, "model_latest.pth") # set to default checkpoint
        print(f"Set checkpoint to default latest model: {args.checkpoint}")
    else:
        out_dir = os.path.dirname(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # %% set up model
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("SYNTRA")
    dkw = 'all' if args.data_keywords is None else args.data_keywords
    log_filename = f"test_{os.path.basename(args.checkpoint).split('.')[0]}_{dkw}.txt"
    logging.basicConfig(filename=join(out_dir, log_filename), level=logging.INFO, filemode="a")

    model = build_segmentation_model(**args).to(device)
    model.eval()

    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    loss_fn = get_loss_fn("UNETloss").to(device)

    assert os.path.isfile(args.checkpoint), f"Checkpoint {args.checkpoint} does not exist."
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    # %% test
    eval_res = single_test(args, device, model, loss_fn, logger, out_dir)
    
    eval_iou = np.array([e[0].item() for e in eval_res]).mean()
    eval_prec = np.array([e[1].item() for e in eval_res]).mean()
    eval_rec = np.array([e[2].item() for e in eval_res]).mean()
    eval_acc = np.array([e[3].item() for e in eval_res]).mean()
    out_str = "#" * 20 + " Overal Mean " + "#" * 20 + "\n" + \
        f"  IoU (FG): {eval_iou*100:.2f}\n" + \
        f"  Precision (FG): {eval_prec*100:.2f}\n" + \
        f"  Recall (FG): {eval_rec*100:.2f}\n" + \
        f"  Accuracy (BG&FG): {eval_acc*100:.2f}\n" + \
        "-" * 40 + "\n"
    logger.info(out_str)
    print(out_str)




if __name__ == "__main__":
    main()
