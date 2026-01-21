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
    parser.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    # model
    parser.add_argument("-r", "--dora-rank", type=int, help="dora rank")
    parser.add_argument("-mn", "--model-name", type=str, help="model name: smol or syntra")
    parser.add_argument("--backbone", type=str, help="backbone type: vit_h, vit_l, vit_b, hiera_t, hiera_s, hiera_l, hiera_bp")
    parser.add_argument("--prompt-encoder", type=str, help="prompt encoder type: example, notion")
    parser.add_argument("--mask-decoder", type=str, help="mask decoder type: default, highres")
    # dataset
    parser.add_argument("-d", "--source-data-dir", type=str, help="path to training datasets")
    parser.add_argument("--num-src", type=int, help="number of source images")
    parser.add_argument("--num-notions", type=int, help="number of notions")

    args = parser.parse_args()

    # print all available hydra config files
    if not GlobalHydra().is_initialized():
        initialize_config_module(config_module="syntra.configs", version_base=None)
    
    cfg = compose(config_name=args.config)['INF']

    # Override cfg arguments with args if provided
    with open_dict(cfg):
        for key, value in vars(args).items():
            if value is not None:
                setattr(cfg, key, value)
    return cfg


class SYNTRA:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.uni_transform = TransformCompose(args.uni_transform) if getattr(args, 'uni_transform', None) is not None else None

        self.model = self.setup_model()
    
    def setup_model(self):
        model = build_segmentation_model(**self.args).to(self.device)
        model.eval()
        # load checkpoint
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device, weights_only=False)
        model.load_trainable_state_dict(checkpoint["model"])
        return model

    def prepare_data(self):
        
        return img_src, lbl_src, img_tgt, aux

    def segment_target(self, source_data, target_data):
        # prepare data
        # load to gpu
        source_data = [d.to(self.device) for d in source_data]
    
        # inference
        with torch.no_grad():
            pred = self.model(img_src, lbl_src, img_tgt, aux)



def main():
    args = parse_args()
    set_seeds(1234)
    syntra = SYNTRA(args)


                

if __name__ == "__main__":
    main()
