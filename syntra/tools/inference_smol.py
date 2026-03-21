# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import os
import sys
import random
import argparse
from tqdm import tqdm
from glob import glob
import json

    
import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2


from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling import build_segmentation_model
from data_tools.indexing.faiss_utils import FAISSIndexer

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
    subparsers = parser.add_subparsers(dest="command", help="sub-commands: inference or test")
    
    # sub-commands for test
    parser_test = subparsers.add_parser("test", help="inference")
    parser_test.add_argument("-d", "--data_root", type=str, help="path to dataset root directory")
    parser_test.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_test.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_test.add_argument("-c", "--config", type=str, help="config file name")
    parser_test.add_argument("-n", "--num-src", type=int, help="number of source images")

    # sub-commands for inference
    parser_inference = subparsers.add_parser("inference", help="inference")
    parser_inference.add_argument("-s", "--source-data-dir", type=str, help="path to source dataset root directory")
    parser_inference.add_argument("-t", "--target-data-dir", type=str, help="path to target dataset root directory")
    parser_inference.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_inference.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_inference.add_argument("-c", "--config", type=str, help="config file name")
    parser_inference.add_argument("-n", "--num-src", type=int, help="number of source images")

    args = parser.parse_args()

    return args


class BaseInferenceDataset(Dataset):
    invalid_cls = ["nonlabeled", "background"]
    def __init__(self, data_root, src_file, ext=".png", normalize=True, img_size=384):
        self.normalize = normalize  
        self.data_root = data_root
        self.subset = os.path.basename(data_root)
        self.tgt_filenames = sorted(os.listdir(os.path.join(data_root, "imgs")))
        src_file = os.path.join(data_root, src_file)
        with open(src_file, 'r') as f:
            self.src_filenames = [line.strip().split(".")[-1] + ext for line in f.readlines()]
        with open(os.path.join(data_root, f"{os.path.basename(data_root)}.json")) as f:
            self.meta_info = json.load(f)
        with open(os.path.join(data_root, "color_map.json")) as f:
            self.color_map = {k: v for k, v in json.load(f).items() if k not in self.invalid_cls}
        
        self.cls_id_to_name = {i+1: name for i, name in enumerate(self.color_map.keys())}
        self.cls_name_to_id = {v:k for k, v in self.cls_id_to_name.items()}
        self.palette = [0, 0, 0] + sum(self.color_map.values(), [])  # add background color

        # Define preprocessing for the image
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),  # Resize 
                transforms.ToTensor(),          # Convert to tensor
                transforms.Normalize(           # Normalize using ImageNet mean and std
                        mean=[0.5, 0.5, 0.5],
                        std=[0.25, 0.25, 0.25]
                    )
            ]
        )

        self._parse_source_samples_for_cls()
        self._parse_file_list()

    def _parse_source_samples_for_cls(self):
        loading_list_src = {c: [] for c in self.color_map.keys()}
        for src_filename in self.src_filenames:
            full_name = f"{self.subset}.{src_filename.split('.')[0]}"
            for cls, ratio in self.meta_info[full_name].items():
                if cls in self.invalid_cls:
                    continue
                if 0.1 < ratio < 0.9:
                    loading_list_src[cls].append(src_filename)
        
        self.loading_list = loading_list_src

    def _parse_file_list(self):
        self.file_list = []
        for tgt_filename in self.tgt_filenames:
            # select one source for each class
            for cls, sources in self.loading_list.items():
                self.file_list.append((tgt_filename, np.random.choice(sources), cls))

    def __len__(self):
        return len(self.file_list)
    
    def load_img(self, filename):
        path = os.path.join(self.data_root, "imgs", filename)
        img = np.array(Image.open(path).convert("RGB"))
        return img
    
    def load_lbl(self, filename, cls):
        path = os.path.join(self.data_root, "lbls", filename)
        lbl = np.array(Image.open(path))
        return lbl == self.cls_name_to_id[cls]
    
    def __getitem__(self, idx):
        tgt_filename, src_filename, cls = self.file_list[idx]
        tgt_img = self.load_img(tgt_filename)
        tgt_lbl = self.load_lbl(tgt_filename, cls)
        src_img = self.load_img(src_filename)
        src_lbl = self.load_lbl(src_filename, cls)

        meta = {
            "src": src_filename,
            "tgt": tgt_filename,
            "cls": cls
        }

        return src_img, src_lbl, tgt_img, tgt_lbl, meta
    
    @staticmethod
    def collate_fn(batch):
        src_imgs, src_lbls, tgt_imgs, tgt_lbls, meta = zip(*batch)
        def _stack_to_tensor(x):
            return torch.from_numpy(np.stack(x, axis=0)).float()
        src_imgs = _stack_to_tensor(src_imgs)
        src_lbls = _stack_to_tensor(src_lbls)
        tgt_imgs = _stack_to_tensor(tgt_imgs)
        tgt_lbls = _stack_to_tensor(tgt_lbls)
        
        return src_imgs, src_lbls, tgt_imgs, tgt_lbls, meta


def build_source_dataloader(data_root, src_file, batch_size=32, num_workers=4, ext='.png', normalize=True):
    dataset = BaseInferenceDataset(data_root, src_file, ext=ext, normalize=normalize)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    assert batch_size % len(dataset.color_map) == 0, "batch size must be multiple of the number of classes."
    return dataloader
    

class SMOLInf:
    def __init__(self, model_name, backbone, dora_rank, mask_decoder, checkpoint, uni_transform, **kwargs):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build transforms
        self.uni_transform = TransformCompose(uni_transform) 

        # build model
        checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = build_segmentation_model(
            model_name=model_name,
            backbone=backbone,
            dora_rank=dora_rank,
            mask_decoder=mask_decoder,
        )
        self.model.load_trainable_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

    def prepare_data(self, source_images, source_labels, target_images):
        # normalize
        source_images = (source_images.float() / 255 - 0.5) / 0.25
        target_images = (target_images.float() / 255 - 0.5) / 0.25

        # load to gpu
        img_src = source_images.permute(0, 3, 1, 2).unsqueeze(1).to(self.device)  # (B, 1, 3, H, W)
        lbl_src = source_labels[:, ::4, ::4].unsqueeze(1).unsqueeze(1).to(self.device)  # (B, 1, 1, H/4, W/4)
        img_tgt = target_images.permute(0, 3, 1, 2).to(self.device)  # (B, 3, H, W)

        return img_src, lbl_src, img_tgt

    def segment_target(self, source_images, source_labels, target_images):
        # input format for the model:
        # img_src: (B, T-1, 3, H, W)
        # lbl_src: (B, T-1, N, H/4, W/4)
        # img_tgt: (B, 3, H, W)
        # prepare data
        img_src, lbl_src, img_tgt = self.prepare_data(
            source_images, source_labels, target_images
        )
        # forward pass
        with torch.no_grad():
            pred = self.model(img_src, lbl_src, img_tgt).sigmoid().squeeze(0).squeeze(1)  # (N, H, W)

        return pred.cpu()
    

def cal_iou(pred_binary_mask, target_binary_mask):
    intersection = torch.logical_and(pred_binary_mask, target_binary_mask).sum().item()
    union = torch.logical_or(pred_binary_mask, target_binary_mask).sum().item()
    iou = intersection / union if union != 0 else 1.0
    return iou

def cal_acc(pred_mask, target_mask):
    correct = (pred_mask == target_mask).sum()
    total = target_mask.size
    acc = correct / total
    return acc


def load_config(config_name: str):
    # Initialize Hydra and compose the configuration
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module="syntra.configs", version_base=None)
    cfg = compose(config_name=config_name)
    return cfg


def inference(args, cfg):
    # ensure result directory
    if args.output_dir is None: 
        args.output_dir = os.path.dirname(args.checkpoint)

    # result_dir = os.path.join(args.output_dir, 'inf_results')
    result_dir = "logs/tmp"
    results = {}
    os.makedirs(result_dir, exist_ok=True)
    # build inference data loader
    dataloader = build_source_dataloader(args.data_root, args.src_file)
    color_map = dataloader.dataset.color_map
    palette = dataloader.dataset.palette

    # get map sheet dimension and result placeholder
    xys = [n.split(".")[0].split("_") for n in dataloader.dataset.tgt_filenames]
    x_max = max([int(xy[0]) for xy in xys]) + 384
    y_max = max(int(xy[1]) for xy in xys) + 384
    map_lbl_pred = Image.new("P", (y_max, x_max))
    map_lbl_pred.putpalette(palette)

    # build model
    smol = SMOLInf(checkpoint=args.checkpoint, **cfg.TRAIN)

    # process subset
    results = {"ious": {k: [] for k in color_map.keys()}, "accs": []}

    for src_imgs, src_lbls, tgt_imgs, tgt_lbls, meta in tqdm(dataloader):
        
        # perform segmentation
        pred_masks = smol.segment_target(src_imgs, src_lbls, tgt_imgs).squeeze(1)

        # evaluate
        pred_binary_masks = pred_masks >= 0.5
        gt_binary_masks = tgt_lbls.bool()
        prob_masks = []
        classes = []
        
        for i, m in enumerate(meta):
            cls_name = m["cls"]
            iou = cal_iou(pred_binary_masks[i], gt_binary_masks[i])
            results["ious"][cls_name].append(iou)

            prob_masks.append(pred_masks[i])
            classes.append(cls_name)
            if (i + 1) % len(color_map) == 0:
                assert classes == [k for k in color_map.keys()]
                pred_segmentation = torch.stack(prob_masks, dim=0).argmax(axis=0) + 1  # add 1 to match class ids
                prob_masks = torch.stack(prob_masks, dim=0)
                pred_segmentation[(prob_masks < 0.5).all(dim=0)] = 0  # set to background if below threshold
                prob_masks = []
                classes = []

                # save results as colored mask
                pred_lbl = Image.fromarray(pred_segmentation.cpu().numpy().astype(np.uint8), mode="P")
                pred_lbl.putpalette(palette)
                # pred_lbl.save(os.path.join(result_dir, m["tgt"]))
                x, y = m["tgt"].split(".")[0].split("_")

                map_lbl_pred.paste(pred_lbl, (int(y), int(x)))

                # map_lbl_pred.save(os.path.join(result_dir, f"{dataloader.dataset.subset}.png"))
    map_lbl_pred.save(os.path.join(result_dir, f"{dataloader.dataset.subset}.png"))

    for cls_name, ious in results["ious"].items():
        print(f"IoU {cls_name} : {np.mean(ious)*100:.2f}")

        

if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--data-root", type=str, help="path to dataset root directory")
    parser.add_argument("-i", "--src-file", type=str, help="filename that list all source samples")
    parser.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser.add_argument("-c", "--config", type=str, help="config file name")

    args = parser.parse_args()
    if len(args.config.split("/")) == 2:
        config_module, args.config = args.config.split("/")
        config_module = "syntra.configs." + config_module
    else:
        config_module = 'syntra.configs'
    initialize_config_module(config_module=config_module, version_base=None)
    cfg = compose(config_name=args.config)

    inference(args, cfg)