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



class SMOLDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, data_root, num_notions=1, num_src=1, mode="train", 
                 info_file=None, fg_ratio=0.05, pos_prob=0.75, filtering_keys=None,
                 src_info_file=None, src_cls=None, **kwargs):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.
        Parameters:
        data_root: str
            The root directory of the dataset containing all related files.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        info_file: str
            Path to the json file containing the data sample information.
        fg_ratio: float, optional
            Foreground ratio threshold for source masks. Default is 0.1.
        pos_prob: float, optional
            Probability of selecting a class that is present in both source and target images. Default is 0.75.
        filtering_keys: str, optional
            Comma-separated keys to filter the dataset files. Default is None.
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        self.fg_ratio = fg_ratio
        self.pos_prob = pos_prob
        
        if mode == "test":
            assert src_info_file is not None, "In test mode, src_info_file must be provided."
            assert src_cls is not None, "In test mode, src_cls must be provided."
            
        self.fixed_cls = src_cls
        self.datasets = os.listdir(self.data_root)
        # load samples
        self.label_ratios = {}
        self.label_colors = {}
        self.file_list = []
        self.src_file_list = []
        available_classes = set()
        for ds in self.datasets:
            if filtering_keys is not None and not any([k in ds for k in filtering_keys]):
                continue
            with open(os.path.join(self.data_root, ds, info_file)) as f:
                lines = list(json.load(f).keys())
                self.file_list.extend(lines)
            with open(os.path.join(self.data_root, ds, f"{ds}.json")) as f:
                self.label_ratios.update(json.load(f))
            with open(os.path.join(self.data_root, ds, f"color_map.json")) as f:
                self.label_colors[ds] = json.load(f)
                available_classes.update(set(self.label_colors[ds].keys()) - set(self.INVALID_CLS))
            if src_info_file is not None:
                with open(os.path.join(self.data_root, ds, src_info_file)) as f:
                    src_lines = list(json.load(f).keys())
                    self.src_file_list.extend(src_lines)
        
        if self.fixed_cls is not None:
            assert self.fixed_cls in available_classes, \
                f"Fixed source class {self.fixed_cls} is not in available classes {available_classes} of the current loaded dataset."

        if len(self.src_file_list) == 0:
            self.src_file_list = self.file_list.copy()

        if filtering_keys is not None:
            filtering_keys = filtering_keys.split(",")
            filtered_file_list = []
            filtered_src_file_list = []
            for fname in self.file_list:
                if any([k in fname for k in filtering_keys]):
                    filtered_file_list.append(fname)
            for fname in self.src_file_list:
                if any([k in fname for k in filtering_keys]):
                    filtered_src_file_list.append(fname)
            self.file_list = filtered_file_list
            self.src_file_list = filtered_src_file_list

        self.file_list = list(set(self.file_list))
        self.src_file_list = list(set(self.src_file_list))
        self.transform = MapTransformTrain() if 'train' in mode or 'nshot' in mode else MapTransformTest()
        print(f"Number of {mode} images: {len(self.file_list)}")
        self.get_loading_list()

    def get_loading_list(self):
        loading_list_src = {}
        for name in self.src_file_list:
            info = self.label_ratios[name]
            key = name.rsplit(".", 1)[0] # subdataset key (map sheet key)
            if key not in loading_list_src:
                loading_list_src[key] = {c: [] for c in info.keys() if c not in self.INVALID_CLS}
            for k, v in info.items(): # k: class, v: ratio
                if k in self.INVALID_CLS:
                    continue
                if self.fg_ratio < v < 0.9 :
                    loading_list_src[key][k].append(name)
        if self.mode == "test":
            file_list = [f for f in self.file_list if len(loading_list_src[f.rsplit(".", 1)[0]][self.fixed_cls]) > 0]
            if len(self.file_list) != len(file_list):
                print(f"Warning: {len(self.file_list) - len(file_list)} target images do not have available source images and will be removed for testing.")
                self.file_list = file_list
        self.loading_list = loading_list_src

    def select_tgt_cls(self, tgt_full_name):
        if 'railway' in tgt_full_name:
            fg_ratio = 0.01
        else:
            fg_ratio = self.fg_ratio

        # available classes of the current target image
        tgt_info = self.label_ratios[tgt_full_name]
        dataset = tgt_full_name.rsplit(".", 1)[0]
        tgt_cls = [k for k, v in tgt_info.items() if v > fg_ratio and k not in self.INVALID_CLS]

        if len(tgt_cls) > 0 and (np.random.random() < self.pos_prob or \
                                 len(tgt_cls) == len(tgt_info) - len(self.INVALID_CLS)):
            # p probability to select a cls that is present in both src and tgt image
            selected_cls = np.random.choice(tgt_cls)
        else:
            # p probability to select a cls that is present in src image but not present in tgt image
            not_tgt_cls = [x for x in tgt_info.keys() if x not in (tgt_cls + self.INVALID_CLS) and len(self.loading_list[dataset][x]) > 0]
            if len(not_tgt_cls) == 0:
                not_tgt_cls = [k for k, v in self.loading_list[dataset].items() if len(v) > 0 and k not in self.INVALID_CLS]
            selected_cls = np.random.choice(not_tgt_cls)

        return selected_cls

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        tgt_full_name = self.file_list[index]
        dataset, _ = tgt_full_name.rsplit(".", 1)

        if self.mode == "train":
            selected_cls = self.select_tgt_cls(tgt_full_name) 
        else:
            selected_cls = self.fixed_cls  
        selected_colors = [self.label_colors[dataset][selected_cls] if selected_cls is not None else None]      
        available_src = self.loading_list[dataset][selected_cls]
        if len(available_src) == 0:
            print(f"Warning: No available source images for target image {tgt_full_name} and selected class {selected_cls}.")
        src_full_names = np.random.choice(available_src, self.num_src, replace=len(available_src)<self.num_src).tolist()           

        # load images and labels
        imgs = []
        lbls = []
        for img_full_name in [tgt_full_name] + src_full_names:
            ds, img_name = img_full_name.rsplit(".", 1)
            img = Image.open(os.path.join(self.data_root, ds, 'imgs', img_name + '.png'))
            img = np.array(img)
            lbl = Image.open(os.path.join(self.data_root, ds, 'lbls', img_name + '.png'))
            # get selected color idx
            palette = lbl.getpalette()
            palette = np.array(palette).reshape(-1, 3).tolist()
            selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
            assert 0 < len(selected_color_inds) <= self.num_notions, \
                f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
            multi_notion_lbl = np.zeros(lbl.size + (self.num_notions,), dtype=bool)
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
    




if __name__ == '__main__':
    dataset = SMOLDataset("/koko/datasets/SMOL_syntra", info_file="train_100shot.txt", mode="train")
    # dataset = SynTraDataset("/koko/datasets/SMOL_syntra", info_file="pairing_train_100shot_train_100shot_intercls.json", mode="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    for d in dataset:
         src_img, src_lbl, tgt_img, tgt_lbl, aux = d
         if src_lbl.sum() == 0 and tgt_lbl.sum() > 0:
             print(src_img.shape, src_lbl.shape, tgt_img.shape, tgt_lbl.shape, aux)
             print("empty src lbl")
             
