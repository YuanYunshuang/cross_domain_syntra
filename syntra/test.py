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
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
import argparse


from syntra.modeling.dataset import get_dataset
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
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-s", "--save-result", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true", default=False)
    # model
    parser.add_argument("-r", "--dora-rank", type=int, default=16)
    parser.add_argument("-mn", "--model-name", type=str, default="smol", help="model name: unet or syntra")
    parser.add_argument("--backbone", type=str, default="vit_b", help="backbone type: vit_h, vit_l, vit_b, hiera_t, hiera_s, hiera_l, hiera_bp")
    parser.add_argument("--prompt-encoder", type=str, default="example", help="prompt encoder type: example, notion")
    parser.add_argument("--mask-decoder", type=str, default="default", help="mask decoder type: default, highres")
    # dataset
    parser.add_argument("-d", "--root-dir", type=str, default=None, help="path to training dataset; two subfolders: lbls and imgs")
    parser.add_argument("-dkw", "--data-keywords", type=str, default=None, help='dataset key words for filtering')
    parser.add_argument("--dataset-name", type=str, default='syntra')
    parser.add_argument("--data-info", type=str, default="pairing_train_100shot_test_intercls.json")
    parser.add_argument("--src-data-info", type=str, default="pairing_train_100shot_train_100shot_intercls.json")
    parser.add_argument("--num-src", type=int, default=1, help="number of source images")
    parser.add_argument("--num-notions", type=int, default=1, help="number of notions")
    parser.add_argument("-fg", "--foreground-ratio", type=float, default=0.1)
    args = parser.parse_args()

    initialize_config_module(config_module="syntra.configs", version_base=None)
    cfg = compose(config_name=args.config)

    # Override args with traincfg arguments
    for key, value in cfg['TRAIN'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    # Override args with test cfg arguments
    for key, value in cfg['TEST'].items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args


def single_test(num_notions, fixed_src_cls, args, device, model, seg_loss, ce_loss, logger, out_dir):
    if args.save_result:
        from syntra.utils.vis import save_results
    if args.debug:
        from syntra.utils.vis import vis_train
    losses = []
    evals = []
    test_dataset = get_dataset(args.dataset_name,
                               data_root=args.root_dir,
                               info_file=args.data_info,
                               src_info_file=args.src_data_info,
                               num_notions=num_notions,
                               num_src=args.num_src,
                               fg_ratio=args.foreground_ratio,
                               filtering_keys=args.data_keywords,
                               mode="test",
                               src_cls=fixed_src_cls,)
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

    evaluator = [Evaluator() for _ in range(num_notions)]

    title = f"Testing {args.data_keywords} {'' if fixed_src_cls is None else fixed_src_cls}"
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

            model_pred = model(img_src, lbl_src, img_tgt)
            lbl_pred = model_pred.sigmoid()
            loss = (seg_loss(model_pred, lbl_tgt.unsqueeze(2)) +
                    ce_loss(model_pred, lbl_tgt.float().unsqueeze(2)))
            for i in range(num_notions):
                evaluator[i].add_sample(lbl_pred.squeeze(2)[:, i], lbl_tgt[:, i], meta=meta)

            pbar.set_postfix({ 'Loss': f'{loss:.6f}'})
            if args.save_result:
                cur_dataset = meta[0]['tgt_name'].rsplit('.', 1)[0]
                color_map = test_dataset.label_colors[cur_dataset]
                save_results(os.path.join(out_dir, f"{str(args.data_keywords)}"), 
                                img_src, lbl_src, img_tgt, lbl_tgt, model_pred, 
                                meta, color_map=color_map)
            losses.append(loss.item())

    avg_loss = np.array(losses).mean()
    eval_out = []
    out_info = f"Average Loss: {avg_loss:.6f}\n" + "=" * 20 +\
        " Evaluation Results " + "=" * 20 + "\n"
    for i in range(num_notions):
        avg_eval_dict = evaluator[i].eval()
        for notion_name,  avg_eval in avg_eval_dict.items():
            eval_out.append(avg_eval)
            out_info += \
                f"Notion set {i+1}, {notion_name}: \n" + \
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
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.checkpoint)
    device = torch.device(args.device)

    # %% set up model
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("SYNTRA")
    logging.basicConfig(filename=join(out_dir, f"test_{os.path.basename(args.checkpoint).split('.')[0]}.txt"), 
                        level=logging.INFO, filemode="a")

    model = build_segmentation_model(**vars(args)).to(device)
    model.eval()

    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    assert os.path.isfile(args.checkpoint)
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_trainable_state_dict(checkpoint["model"])

    # %% test
    eval_results = []
    if args.dataset_name == 'cd_syntra' or args.dataset_name == 'syntra':
        fixed_src_cls = None
        eval_res = single_test(args.num_notions, fixed_src_cls, args, device, model, seg_loss, ce_loss, logger, out_dir)
        eval_results.extend([torch.stack(x, dim=0) for x in eval_res])
    elif args.dataset_name == 'smol':
        num_notions = 1
        fixed_src_cls = {
            'hameln': ['forest', 'grass', 'settlement', 'water'], # 'forest', 'grass', 'settlement', 
            'bayern': ['forest', 'grass', 'water'],
            'railway': ['railway'],
            'vineyard': ['vineyard']
        }
        for kw, classes in fixed_src_cls.items():
            for cls in classes:
                args.keywords = kw
                eval_res = single_test(num_notions, cls, args, device, model, seg_loss, ce_loss, logger, out_dir)
                eval_results.extend([torch.stack(x, dim=0) for x in eval_res])
    
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
