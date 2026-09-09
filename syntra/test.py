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
from syntra.tools.eval import Evaluator
from syntra.utils.debug_utils import *
from hydra import compose, initialize_config_module

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
    parser.add_argument("--vis", action="store_true", help="save visualization for these test samples")
    parser.add_argument("--debug", action="store_true", default=False)
    # model
    parser.add_argument("-r", "--dora-rank", type=int, )
    parser.add_argument("-mn", "--model-name", type=str, help="model name: unet or syntra")
    parser.add_argument("--backbone", type=str, help="backbone type: vit_h, vit_l, vit_b, hiera_t, hiera_s, hiera_l, hiera_bp")
    parser.add_argument("--prompt-encoder", type=str, help="prompt encoder type: example, notion")
    parser.add_argument("--mask-decoder", type=str,  help="mask decoder type: default, highres")
    # dataset
    parser.add_argument("-d", "--root-dir", type=str, default=None, help="path to training dataset; two subfolders: lbls and imgs")
    parser.add_argument("-dkw", "--data-keywords", type=str, default=None, help='dataset key words for filtering')
    parser.add_argument("--dataset-name", type=str, )
    parser.add_argument("--data-info", type=str, )
    parser.add_argument("--img-size", type=int)
    parser.add_argument("-xkw", "--exclude-keys", type=str,
                        help="comma-separated keys of subsets to drop, e.g. a held-out sheet")
    parser.add_argument("--src-fg-max", type=float,
                        help="drop source prompts the class covers more than this fraction of. "
                             "Must match the value the checkpoint was trained with")
    parser.add_argument("--source-file", type=str,
                        help="SMOL only: sample list the prompts are drawn from. Targets in test "
                             "mode are always <mode>_samples.txt, so this sets the prompt pool "
                             "without changing what is scored")
    parser.add_argument("--target-file", type=str, help="SMOL only: unused in test mode")
    parser.add_argument("--src-selection", type=str,
                        help="source selection: retrieved (default), topk, or random")
    parser.add_argument("--src-fusion", type=str,
                        help="fusion over the source predictions: max or mean")
    parser.add_argument("--num-src", type=int, help="number of source images")
    parser.add_argument("--num-notions", type=int, help="number of notions")
    parser.add_argument("-fg", "--foreground-ratio", type=float, help="minimum foreground ratio in the target image")
    args = parser.parse_args()

    if len(args.config.split("/")) == 2:
        config_module, args.config = args.config.split("/")
        config_module = "syntra.configs." + config_module
    else:
        config_module = 'syntra.configs'
    initialize_config_module(config_module=config_module, version_base=None)
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


def single_test(num_notions, fixed_src_cls, args, device, model, seg_loss, ce_loss, logger, out_dir):
    if args.save_result:
        from syntra.utils.vis import save_results
    if args.vis:
        from syntra.utils.vis import save_results_v2 as save_vis
    losses = []
    uni_transform = TransformCompose(args.uni_transform) if getattr(args, 'uni_transform', None) is not None else None
    test_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               info_file=args.data_info,
                               num_notions=num_notions,
                               num_src=args.num_src,
                               fg_ratio=args.foreground_ratio,
                               filtering_keys=args.data_keywords,
                               mode="test",
                               src_cls=fixed_src_cls,
                               img_size=args.img_size,
                               uni_transform=uni_transform,
                               **({'src_selection': args.src_selection}
                                  if getattr(args, 'src_selection', None) is not None else {}),
                               **({'exclude_keys': args.exclude_keys}
                                  if getattr(args, 'exclude_keys', None) is not None else {}),
                               # SMOL only: names the budgeted sample list its prompts come from.
                               # In test mode the targets are always <mode>_samples.txt, so this
                               # sets the prompt pool without touching what is scored.
                               **({'src_fg_max': args.src_fg_max}
                                  if getattr(args, 'src_fg_max', None) is not None else {}),
                               **({'source_file': args.source_file}
                                  if getattr(args, 'source_file', None) is not None else {}),
                               **({'target_file': args.target_file}
                                  if getattr(args, 'target_file', None) is not None else {}))
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

    evaluator = Evaluator()
    dkw = 'all' if args.data_keywords is None else args.data_keywords
    src_cls_str = '' if fixed_src_cls is None else fixed_src_cls
    title = f" Testing {dkw} {src_cls_str}"
    title = ">" * 20 + title + "<" * 20
    logger.info(title)
    print(title)
    with (torch.no_grad()):
        pbar = tqdm(test_dataloader)
        # pbar.set_postfix({'Progress': f'0/{num_samples}', 'Loss': f'{0:.6f}',
        #                   'IoU': f'{0:.3f}', 'Precision': f'{0:.3f}', 'Recall': f'{0:.3f}'})
        for step, data in enumerate(pbar):
            img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
            meta = data[4]
            if hasattr(test_dataset, 'prepare_inputs'):
                img_src, lbl_src, img_tgt, lbl_tgt, meta = test_dataset.prepare_inputs(
                    img_src, lbl_src, img_tgt, lbl_tgt, meta)

            model_pred = model(img_src, lbl_src, img_tgt)
            lbl_pred = model_pred.sigmoid()
            loss = (seg_loss(model_pred, lbl_tgt.unsqueeze(2)) +
                    ce_loss(model_pred, lbl_tgt.float().unsqueeze(2)))
            evaluator.add_sample(lbl_pred.squeeze(2), lbl_tgt, meta=meta)

            pbar.set_postfix({ 'Loss': f'{loss:.6f}'})

            cur_datasets = [x['tgt_name'].rsplit('.', 1)[0] for x in meta]
            color_maps = [test_dataset.label_colors[cur_dataset] for cur_dataset in cur_datasets]
            if args.save_result:
                save_results(os.path.join(out_dir, f"{str(dkw)}"), 
                                img_src, lbl_src, img_tgt, lbl_tgt, model_pred, 
                                meta, color_maps=color_maps)
            if args.vis:
                save_vis(os.path.join(out_dir, f"vis_{str(dkw)}"), 
                                img_src, lbl_src, img_tgt, lbl_tgt, model_pred, 
                                meta, color_maps=color_maps)
            losses.append(loss.item())

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
        # -o wins when given. It used to be ignored whenever -w was passed, so a sweep over
        # inference settings on one checkpoint wrote every variant into that checkpoint's own
        # result file - and the logger opens in append mode, so they stacked up as extra blocks
        # behind the canonical number. That is exactly what makes the archived SMOL files hard to
        # read: smol_hismap_n1_s100 holds two runs at two different test-set sizes, and quoting
        # the wrong block flatters the baseline by four points.
        out_dir = args.output_dir or os.path.dirname(args.checkpoint)

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
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    assert os.path.isfile(args.checkpoint), f"Checkpoint {args.checkpoint} does not exist."
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_trainable_state_dict(checkpoint["model"])

    # %% test
    eval_results = []
    fixed_src_cls = None
    eval_res = single_test(args.num_notions, fixed_src_cls, args, device, model, seg_loss, ce_loss, logger, out_dir)
    eval_results.extend([torch.stack(x, dim=0) for x in eval_res])
    # if 'syntra' in args.dataset_name:
    #     fixed_src_cls = None
    #     eval_res = single_test(args.num_notions, fixed_src_cls, args, device, model, seg_loss, ce_loss, logger, out_dir)
    #     eval_results.extend([torch.stack(x, dim=0) for x in eval_res])
    # elif args.dataset_name == 'smol':
    #     num_notions = 1
    #     fixed_src_cls = {
    #         'hameln': ['forest', 'grass', 'settlement', 'water'], # 'forest', 'grass', 'settlement', 
    #         'bayern': ['forest', 'grass', 'water'],
    #         'railway': ['railway'],
    #         'vineyard': ['vineyard']
    #     }
    #     for kw, classes in fixed_src_cls.items():
    #         for cls in classes:
    #             args.keywords = kw
    #             eval_res = single_test(num_notions, cls, args, device, model, seg_loss, ce_loss, logger, out_dir)
    #             eval_results.extend([torch.stack(x, dim=0) for x in eval_res])
    
    eval_results = torch.stack(eval_results, dim=0).mean(dim=0)
    out_str = "#" * 20 + " Overal Mean " + "#" * 20 + "\n" + \
        f"  IoU (FG): {eval_results[0]*100:.2f}\n" + \
        f"  Precision (FG): {eval_results[1]*100:.2f}\n" + \
        f"  Recall (FG): {eval_results[2]*100:.2f}\n" + \
        f"  Accuracy (BG&FG): {eval_results[3]*100:.2f}\n" + \
        "-" * 40 + "\n"
    logger.info(out_str)
    print(out_str)




if __name__ == "__main__":
    main()
