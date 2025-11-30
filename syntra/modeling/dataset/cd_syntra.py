from glob import glob
import os

from PIL import Image
from tifffile import TiffFile, memmap
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from typing import List
from syntra.modeling.dataset.trainsform import MapTransformTrain, MapTransformTest


class CrossDomainSynTraDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, data_root, info_file, num_notions=1, num_src=1, mode="train", 
                 fg_ratio=0.05, pos_prob=0.9, filtering_keys=None, **kwargs):
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

        # load samples
        self.src_pool = {}
        self.label_ratios = {}
        self.label_colors = {}
        self.meta_info = {}
        self.dataset_full_paths = {}
        self.filtering_keys = [x.strip() for x in filtering_keys.split(",")] if filtering_keys is not None else None

        # parse info files
        if isinstance(info_file, str):
            datasets = glob(os.path.join(self.data_root, f"**/{info_file}.json"), recursive=True)
            datasets = [os.path.dirname(ds) for ds in datasets]
            # unified info file for all datasets
            info_file_dict = {info_file: datasets}
        else:
            try:
                info_file = dict(info_file)
            except:
                raise ValueError("info_file should be either a str or a dict mapping info_file to dataset list.")
            info_file_dict = {info_file_name: [os.path.join(self.data_root, x) for x in ds_list] for info_file_name, ds_list in info_file.items()}

        for info_file, datasets in info_file_dict.items():
            info_file += '.json'
            for ds_path in datasets:
                ds = os.path.basename(ds_path)
                self.dataset_full_paths[ds] = ds_path
                if self.filtering_keys is not None and not any([k in ds_path for k in self.filtering_keys]):
                    continue
                if not os.path.isfile(os.path.join(ds_path, info_file)):
                    Warning(f"Info file {info_file} does not exist in dataset folder {ds_path}. Skip this dataset.")
                    continue
                with open(os.path.join(ds_path, info_file)) as f:
                    self.src_pool.update({f"{k}.{ds}": v for k, v in json.load(f).items() if k!='negative'})
                with open(os.path.join(ds_path, f"{ds}.json")) as f:
                    self.label_ratios.update(json.load(f))
                with open(os.path.join(ds_path, f"color_map.json")) as f:
                    self.label_colors[ds] = {k: v for k, v in json.load(f).items() if k not in self.INVALID_CLS}
                if os.path.isfile(os.path.join(ds_path, "meta_info.json")):
                    with open(os.path.join(ds_path, "meta_info.json")) as f:
                        self.meta_info[ds] = json.load(f)

        self._parse_samples()

        self.transform = MapTransformTrain() if 'train' in mode else MapTransformTest()
        assert len(self.file_list) > 0, f"No samples found for mode {mode} with filtering keys {self.filtering_keys}."
        print(f"Number of {mode} images: {len(self.file_list)}")
    
    def _parse_samples(self):
        self.file_list = []
        pos_file_list = []
        neg_file_list = []
        removed_count = 0
        for cls_full_name, samples in self.src_pool.items():
            cls_name = cls_full_name.split(".")[0]
            for sample_name, sample_info in samples.items():
                if len(sample_info['src_samples']) == 0:
                    removed_count += 1
                    continue
                if self.label_ratios[sample_name].get(cls_name, 0.0) > self.fg_ratio:
                    pos_file_list.append(f"{cls_full_name}/{sample_name}")
                else:
                    neg_file_list.append(f"{cls_full_name}/{sample_name}")
        if removed_count > 0:
            print(f"Removed {removed_count} samples with no source examples.")
        # pos/neg sample ratio 0.8:0.2
        num_pos = len(pos_file_list) 
        num_neg = int(num_pos * (1 - self.pos_prob) / self.pos_prob)
        self.file_list = pos_file_list + random.sample(neg_file_list, min(num_neg, len(neg_file_list)))
    
    def __len__(self):
        return len(self.file_list)
    
    def _select_src_samples(self, cls_name, tgt_full_name):
        src_samples = self.src_pool[cls_name][tgt_full_name]['src_samples']
        assert len(src_samples) > 0, f"No source samples found for target image {tgt_full_name} and class {cls_name}."
        src_full_names = np.random.choice(
            src_samples, 
            self.num_src, 
            replace=len(src_samples)<self.num_src
        ).tolist()
        return src_full_names
    
    def _load_image_label(self, full_name):
        ds, img_name = full_name.rsplit(".", 1)
        meta = self.meta_info.get(ds, None)
        img_ext = meta.get("image_format", "png") if meta is not None else "png"
        img_path = os.path.join(self.dataset_full_paths[ds], 'imgs', img_name + f'.{img_ext}')
        lbl_ext = meta.get("label_format", "png") if meta is not None else "png"
        lbl_path = os.path.join(self.dataset_full_paths[ds], 'lbls', img_name + f'.{lbl_ext}')
        
        if img_ext == 'tiff' or img_ext == 'tif':
            img = Image.open(img_path)
            img = np.array(img)
        else:
            img = Image.open(img_path)
            img = np.array(img)
        if img.ndim == 2:
            img = img[..., np.newaxis].repeat(3, axis=-1)
        
        lbl = Image.open(lbl_path)
        return img, lbl
    
    def _select_notions(self, cls_full_name, dataset, src_full_names):
        cls_name = cls_full_name.split(".")[0]
        if self.num_notions == 1:
            selected_classes = [cls_name]
            selected_colors = [self.label_colors[dataset][cls_name]]
            return selected_classes, selected_colors
        else:
            # in addition to the main class, randomly select other classes from the target image to improve diversity
            cls_ratios = {}
            for src_full_name in src_full_names:
                for cls, ratio in self.label_ratios[src_full_name].items():
                    if ratio == 0.0:
                        continue
                    if cls not in cls_ratios:
                        cls_ratios[cls] = 0.0
                    cls_ratios[cls] += ratio
            # select classes based on the max ratios
            sorted_cls = sorted(cls_ratios.items(), key=lambda x: x[1], reverse=True)
            selected_classes = [cls_name]
            for cls, _ in sorted_cls:
                if cls != cls_name and cls in self.label_colors[dataset]:
                    selected_classes.append(cls)
                if len(selected_classes) >= self.num_notions:
                    break
            selected_colors = [self.label_colors[dataset].get(c, None) for c in selected_classes]
        return selected_classes, selected_colors
    
    def __getitem__(self, index):
        cls_name, tgt_full_name = self.file_list[index].split("/")
        dataset, img_name = tgt_full_name.rsplit(".", 1)

        src_full_names = self._select_src_samples(cls_name, tgt_full_name)

        selected_classes, selected_colors = self._select_notions(cls_name, dataset, src_full_names)               

        # load images and labels
        imgs = []
        lbls = []
        # uniq_lbls = []
        for img_full_name in [tgt_full_name] + src_full_names:
            # ds, img_name = img_full_name.rsplit(".", 1)
            img, lbl = self._load_image_label(img_full_name)
            # get selected color idx
            palette = lbl.getpalette()
            palette = np.array(palette).reshape(-1, 3).tolist()
            selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
            assert 0 < len(selected_color_inds) <= self.num_notions, \
                f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
            multi_notion_lbl = np.zeros((lbl.size[1], lbl.size[0], self.num_notions,), dtype=np.uint8)
            lbl = np.array(lbl.convert("P"))
            # uniq_lbls.append(np.unique(lbl).tolist())
            for i, color_idx in enumerate(selected_color_inds):
                if color_idx is not None:
                    multi_notion_lbl[..., i][lbl == color_idx] = 1
                # add the ignore mask
                multi_notion_lbl[..., i][lbl == 255] = 255
            imgs.append(img)
            lbls.append(multi_notion_lbl)

        # print(cls_name, uniq_lbls, selected_color_inds)
        # self.visualize(img, lbl)
        imgs, lbls = self.transform(imgs, lbls)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)
        tgt_lbl = lbls[0]
        src_lbls = src_lbls[..., ::4, ::4]
        # tgt_lbl = tgt_lbl[::4, ::4]
        full_cls_names = [f"{dataset}/{x}" for x in selected_classes] 
        aux = {'src_name': src_full_names, 'tgt_name': tgt_full_name, 'cls_name': full_cls_names, 'cls_color': selected_colors}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            cls_full_name = fname.split('/')[0]
            loc = '.'.join(cls_full_name.split('.')[:2])  
            return loc
        fname = self.file_list[index]
        loc = get_loc(fname)
        # Map loc strings to stable ids [0..K-1]
        if not hasattr(self, "_loc2id"):
            # build mapping once
            uniq: List[str] = sorted({get_loc(f) for f in self.file_list})
            self._loc2id = {v: i for i, v in enumerate(uniq)}
            print(f"Sampling groups: {self._loc2id}")
        return self._loc2id[loc]

    @staticmethod
    def collate_fn(batch):
        src_img, src_lbl, tgt_img, tgt_lbl, aux = zip(*batch)
        src_img = torch.stack(src_img, dim=0)
        src_lbl = torch.stack(src_lbl, dim=0)
        tgt_img = torch.stack(tgt_img, dim=0)
        tgt_lbl = torch.stack(tgt_lbl, dim=0)
        
        return src_img, src_lbl, tgt_img, tgt_lbl, aux

