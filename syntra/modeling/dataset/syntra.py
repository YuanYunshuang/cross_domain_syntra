from glob import glob
import os

from PIL import Image
from tifffile import imread
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from typing import List
from .trainsform import MapTransformTrain, MapTransformTest


class SynTraDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, data_root, info_file, num_notions=1, num_src=1, mode="train", 
                 fg_ratio=0.05, pos_prob=0.75, filtering_keys=None, **kwargs):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.

        Parameters:
        data_root: str
            The root directory of the dataset containing all related files.
        info_file: str
            Path to the json file containing the dataset information.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        fg_ratio: float, optional
            Foreground ratio threshold for source masks. Default is 0.1.
        pos_prob: float, optional
            Probability of selecting a class that is present in both source and target images. Default is 0.75.
        filtering_keys: str, optional
            Comma-separated keys to filter the sub-datasets that contain the given keys. Default is None to select all.
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        self.fg_ratio = fg_ratio
        self.pos_prob = pos_prob

        datasets = os.listdir(self.data_root)
        # load samples
        self.src_pool = {}
        self.label_ratios = {}
        self.label_colors = {}
        self.meta_info = {}
        
        for ds in datasets:
            if filtering_keys is not None and not any([k in ds for k in filtering_keys.split(",")]):
                continue
            if not os.path.isfile(os.path.join(self.data_root, ds, info_file)):
                Warning(f"Info file {info_file} does not exist in dataset folder {ds}. Skip this dataset.")
                continue
            with open(os.path.join(self.data_root, ds, info_file)) as f:
                self.src_pool.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"{ds}.json")) as f:
                self.label_ratios.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"color_map.json")) as f:
                self.label_colors[ds] = {k: v for k, v in json.load(f).items() if k not in self.INVALID_CLS}
            if os.path.isfile(os.path.join(self.data_root, ds, "meta_info.json")):
                with open(os.path.join(self.data_root, ds, "meta_info.json")) as f:
                    self.meta_info[ds] = json.load(f)

        self.file_list = list(self.src_pool.keys())

        self.transform = MapTransformTrain() if 'train' in mode or 'nshot' in mode else MapTransformTest()
        print(f"Number of {mode} images: {len(self.file_list)}")

    def select_tgt_cls(self, tgt_full_name, src_full_names):
        if self.mode == "test":
            fg_ratio = 0.0
        elif 'railway' in tgt_full_name:
            fg_ratio = 0.01
        else:
            fg_ratio = self.fg_ratio
        # parse available classes of the current target image
        tgt_info = self.label_ratios[tgt_full_name]
        tgt_available_classes = [k for k, v in tgt_info.items() if v > fg_ratio and k not in self.INVALID_CLS]
        tgt_available_classes = set(tgt_available_classes)

        # parse available classes of the current source images
        src_available_classes = []
        for src_full_name in src_full_names:
            src_info = self.label_ratios[src_full_name]
            src_available_classes.extend([k for k, v in src_info.items() if v > fg_ratio and k not in self.INVALID_CLS])
        src_available_classes = set(src_available_classes)

        # cal sample probability: p to select a class that is present in both src and tgt image
        # prob 1-p to select a class that is present in src image but not present in tgt image
        available_classes = list(src_available_classes)
        pos_classes = list(src_available_classes & tgt_available_classes)
        neg_classes = list(src_available_classes - tgt_available_classes)
        
        if len(available_classes) == 0:
            Warning(f"No available classes for target image {tgt_full_name} and source images {src_full_names}. Return None.")
            return None
        
        if random.random() < self.pos_prob and len(pos_classes) > 0:
            selected_cls = np.random.choice(pos_classes)
        else:
            if len(neg_classes) == 0:
                neg_classes = pos_classes
            selected_cls = np.random.choice(neg_classes)
        
        return selected_cls

    def _select_src_samples(self, tgt_full_name):
        src_full_names = self.src_pool[tgt_full_name]['src_samples']
        src_dists = self.src_pool[tgt_full_name]['distances']
        if self.mode == "train":
            # random select num_src from the available src images
            src_full_names = np.random.choice(src_full_names, self.num_src, replace=len(src_full_names)<self.num_src).tolist()
        else:
            # select top-k closest src images
            sorted_indices = np.argsort(src_dists)
            src_full_names = [src_full_names[i] for i in sorted_indices[:self.num_src]]
        return src_full_names
    
    def _load_image_label(self, full_name):
        ds, img_name = full_name.rsplit(".", 1)
        meta = self.meta_info.get(ds, None)
        img_ext = meta.get("image_format", "png") if meta is not None else "png"
        lbl_ext = meta.get("label_format", "png") if meta is not None else "png"
        if img_ext == 'tiff' or img_ext == 'tif':
            img = imread(os.path.join(self.data_root, ds, 'imgs', img_name + f'.{img_ext}'))
        else:
            img = Image.open(os.path.join(self.data_root, ds, 'imgs', img_name + f'.{img_ext}'))
            img = np.array(img)
        if img.ndim == 2:
            img = img[..., np.newaxis].repeat(3, axis=-1)
        
        lbl = Image.open(os.path.join(self.data_root, ds, 'lbls', img_name + f'.{lbl_ext}'))
        return img, lbl


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_full_name = self.file_list[index]
        dataset, _ = tgt_full_name.rsplit(".", 1)
        src_full_names = self._select_src_samples(tgt_full_name)

        if self.mode == "train":
            selected_cls = self.select_tgt_cls(tgt_full_name, src_full_names) 
            selected_colors = [self.label_colors[dataset][selected_cls] if selected_cls is not None else None]                 
        else:
            # for test mode, use all notions
            selected_colors = [v for k, v in self.label_colors[dataset].items() if k not in self.INVALID_CLS]
            assert len(selected_colors) <= self.num_notions, \
                f"During test, number of selected notions {len(selected_colors)} should be == {self.num_notions}."

        # load images and labels
        imgs = []
        lbls = []
        for img_full_name in [tgt_full_name] + src_full_names:
            ds, img_name = img_full_name.rsplit(".", 1)
            img, lbl = self._load_image_label(img_full_name)
            # get selected color idx
            palette = lbl.getpalette()
            palette = np.array(palette).reshape(-1, 3).tolist()
            selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
            assert 0 < len(selected_color_inds) <= self.num_notions, \
                f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
            multi_notion_lbl = np.zeros((lbl.size[1], lbl.size[0], self.num_notions,), dtype=bool)
            lbl = np.array(lbl.convert("P"))
            for i, color_idx in enumerate(selected_color_inds):
                if color_idx is not None:
                    multi_notion_lbl[..., i] = lbl == color_idx
            imgs.append(img)
            lbls.append(multi_notion_lbl)

        # self.visualize(img, lbl)
        imgs, lbls = self.transform(imgs, lbls)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)
        tgt_lbl = lbls[0]
        src_lbls = src_lbls[..., ::4, ::4]
        # tgt_lbl = tgt_lbl[::4, ::4]
        aux = {'src_name': src_full_names, 'tgt_name': tgt_full_name}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            if 'siegfried' in fname:
                loc = fname.split(".")[1].split('_')[0]
            else:
                loc = fname.split('.')[0]
            return loc
        fname = self.file_list[index]
        loc = get_loc(fname)
        # Map loc strings to stable ids [0..K-1]
        if not hasattr(self, "_loc2id"):
            # build mapping once
            uniq: List[str] = sorted({get_loc(f) for f in self.file_list})
            self._loc2id = {v: i for i, v in enumerate(uniq)}
        return self._loc2id[loc]

    @staticmethod
    def collate_fn(batch):
        src_img, src_lbl, tgt_img, tgt_lbl, aux = zip(*batch)
        src_img = torch.stack(src_img, dim=0)
        src_lbl = torch.stack(src_lbl, dim=0)
        tgt_img = torch.stack(tgt_img, dim=0)
        tgt_lbl = torch.stack(tgt_lbl, dim=0)
        
        return src_img, src_lbl, tgt_img, tgt_lbl, aux
    