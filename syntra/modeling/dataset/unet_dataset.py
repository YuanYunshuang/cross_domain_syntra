# Dataset and dataloader for UNet model training

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from glob import glob
from typing import List, Dict, Optional

from syntra.modeling.dataset.trainsform import MapTransformTrain, MapTransformTest


class UNetDataset(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, 
                 data_root : str, 
                 info_file : str | Dict[str, List[str]],
                 img_size : int = 384, 
                 mode : str = "train", 
                 filtering_keys : str = None, 
                 uni_transform: Optional[object] = None,
                 aug_transform: Optional[object] = None,
                 **kwargs):
        """
        Initializes an instance of the class with dataset information, loads file lists,
        and initializes transformations.

        Parameters:
        data_root: str
            The root directory of the dataset containing all related files.
        info_file: str
            Path to the json file containing the dataset information.
        img_size: int, optional
            The size to which images will be resized. Default is 384.
        mode: str
            Indicates the mode in which the dataset is being used. Default is "train".
        filtering_keys: str, optional
            Comma-separated keys to filter the sub-datasets that contain the given keys. Default is None to select all.
        uni_transform: object, optional
            Unified Transformations to be applied to the images and labels. Such as ToTensor, Normalize, etc., 
            which are not related to data augmentation.
            Default is MapTransformTrain for train and MapTransformTest for test mode.
        aug_transform: object, optional
            Augmentation Transformations to be applied to the images and labels. Such as RandomRotate, RandomFlip, etc.
            Default is MapTransformTrain which contains random rotate and flip.
        """
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size
        self.filtering_keys = [x.strip() for x in filtering_keys.split(",")] if filtering_keys is not None else None
        self.color_map = {}
        self.samples = []

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

        # parse samples
        for info_file, datasets in info_file_dict.items():
            info_file += '.json'
            for ds_path in datasets:
                ds = os.path.basename(ds_path)
                
                if self.filtering_keys is not None and not any([k in ds_path for k in self.filtering_keys]):
                    continue
                if not os.path.isfile(os.path.join(ds_path, info_file)):
                    Warning(f"Info file {info_file} does not exist in dataset folder {ds_path}. Skip this dataset.")
                    continue
                with open(os.path.join(ds_path, info_file)) as f:
                    src_dict = json.load(f)
                with open(os.path.join(ds_path, f"color_map.json")) as f:
                    color_map = {k: v for k, v in json.load(f).items() if k not in self.INVALID_CLS}
                    for k, v in color_map.items():
                        if k not in self.color_map:
                            self.color_map[k] = v
                new_cls_names = [k for k in self.color_map.keys()]
                old_to_new_id = {i+1: new_cls_names.index(k) + 1 for i, k in enumerate(color_map.keys())}
                # parse sample paths
                for sample in src_dict.keys():
                    sample_name = sample.split('.')[-1]
                    self.samples.append((
                        os.path.join(ds_path, 'imgs', sample_name + ".png"),
                        os.path.join(ds_path, 'lbls', sample_name + ".png"),
                        old_to_new_id,
                    ))

        self.uni_transform = uni_transform
        self.aug_transform = aug_transform
        if self.uni_transform is None:
            self.uni_transform = MapTransformTrain(image_size=self.img_size) if 'train' in mode else MapTransformTest(image_size=self.img_size)
        
        print(f"Number of {mode} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, lbl_path, id_map = self.samples[idx]
        subset = img_path.split('/')[-3]
        img = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        lbl = Image.open(os.path.join(self.data_root, lbl_path)) 
        lbl = np.array(lbl)
        # remap label ids
        new_lbl = np.zeros_like(lbl)
        for old_id, new_id in id_map.items():
            new_lbl[lbl == old_id] = new_id
        
        lbl = Image.fromarray(new_lbl)
        
        img, lbl = self.uni_transform(img, lbl)

        return img, lbl, subset
    
    def collate_fn(self, batch):
        imgs = []
        lbls = []
        subsets = []
        for b in batch:
            imgs.append(b[0])
            lbls.append(b[1])
            subsets.append(b[2])
        imgs = torch.stack(imgs, axis=0)
        lbls = torch.stack(lbls, axis=0)
        return imgs, lbls, subsets
    
    def prepare_inputs(self, imgs, lbls):
        if self.aug_transform is not None:
            imgs, lbls = self.aug_transform(imgs, lbls)
        return imgs, lbls
    
    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            loc = fname.split('/')[-3]
            if 'railway' in loc.lower():
                loc = 'railway'
            elif 'vineyard' in loc.lower():
                loc = 'vineyard'
            else:
                loc = loc.split('.')[0]
            return loc
        fname = self.samples[index][0]
        loc = get_loc(fname)
        # Map loc strings to stable ids [0..K-1]
        if not hasattr(self, "_loc2id"):
            # build mapping once
            uniq: List[str] = sorted({get_loc(f[0]) for f in self.samples})
            self._loc2id = {v: i for i, v in enumerate(uniq)}
            print(f"Sampling groups: {self._loc2id}")
        return self._loc2id[loc]
    
    def save_train_vis(self, imgs, lbls, preds):
        save_path = "/home/yuan/Downloads/vis_train/tmp.png"
        palette = [0, 0, 0] + [v for k, v in self.color_map.items() for v in v]
        B, C, H, W = imgs.shape
        
        vis_img = Image.new("RGB", (W * 3, H))
        img = imgs[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.25 + 0.5).astype(np.uint8)

        lbl = lbls[0].squeeze(0).cpu().numpy().astype(np.uint8)
        pred = torch.argmax(preds[0], dim=0).cpu().numpy().astype(np.uint8)

        vis_img.paste(Image.fromarray(img), (0, 0))
        lbl_img = Image.fromarray(lbl, mode='P')
        lbl_img.putpalette(palette)
        vis_img.paste(lbl_img.convert("RGB"), (W, 0))
        pred_img = Image.fromarray(pred, mode='P')
        pred_img.putpalette(palette)
        vis_img.paste(pred_img.convert("RGB"), (W * 2, 0))
        vis_img.save(save_path)