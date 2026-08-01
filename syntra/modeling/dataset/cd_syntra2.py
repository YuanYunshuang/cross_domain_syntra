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
from .trainsform import MapTransformTrain, MapTransformTest
from typing import Dict, Optional


class CrossDomainSynTraDataset2(Dataset):
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, 
                 data_root : str, 
                 info_file : str | Dict[str, List[str]],
                 img_size : int = 384, 
                 num_notions : int = 1, 
                 num_src : int = 1, 
                 mode : str = "train", 
                 fg_ratio : float | Dict[str, float] = 0.05,
                 pos_prob : float = 0.8,
                 filtering_keys : str = None,
                 uni_transform: Optional[object] = None,
                 aug_transform: Optional[object] = None,
                 src_selection : str = "retrieved",
                 exclude_keys : str = None,
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
        fg_ratio: float, optional
            Foreground ratio threshold for source masks. Default is 0.1.
        pos_prob: float, optional
            Probability of selecting a class that is present in both source and target images. Default is 0.75.
        filtering_keys: str, optional
            Comma-separated keys to filter the sub-datasets that contain the given keys. Default is None to select all.
        uni_transform: object, optional
            Unified Transformations to be applied to the images and labels. Such as ToTensor, Normalize, etc., 
            which are not related to data augmentation.
            Default is MapTransformTrain for train and MapTransformTest for test mode.
        aug_transform: object, optional
            Augmentation Transformations to be applied to the images and labels. Such as RandomRotate, RandomFlip, etc.
            Default is MapTransformTrain which contains random rotate and flip.
        src_selection: str, optional
            How the num_src source examples are drawn: "retrieved" (uniformly among the
            retrieved neighbours of the target, the default and the behaviour of all earlier
            runs), "topk" (the closest neighbours in ranking order) or "random" (uniformly from
            all annotated tiles of the same subset and class, i.e. retrieval disabled).
        exclude_keys: str, optional
            Comma-separated keys; any subset whose path contains one of them is dropped, after
            filtering_keys has been applied. Used to hold subsets out of training and to drop
            the duplicated scan (see docs/DATA_ISSUE_hameln_3921_duplicate.md).
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        assert src_selection in ("retrieved", "topk", "random"), \
            f"Unknown source selection: {src_selection}"
        self.src_selection = src_selection
        self.fg_ratio = {}
        self.pos_prob = pos_prob
        self.img_size = img_size

        # load samples
        self.src_pool = {}
        self.label_ratios = {}
        self.label_colors = {}
        self.meta_info = {}
        self.dataset_full_paths = {}
        self.filtering_keys = [x.strip() for x in filtering_keys.split(",")] if filtering_keys is not None else None
        self.exclude_keys = [x.strip() for x in exclude_keys.split(",")] if exclude_keys else None

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
                
                if self.filtering_keys is not None and not any([k in ds_path for k in self.filtering_keys]):
                    continue
                if self.exclude_keys is not None and any([k in ds_path for k in self.exclude_keys]):
                    print(f"Excluding subset {ds}")
                    continue
                if not os.path.isfile(os.path.join(ds_path, info_file)):
                    Warning(f"Info file {info_file} does not exist in dataset folder {ds_path}. Skip this dataset.")
                    continue
                with open(os.path.join(ds_path, info_file)) as f:
                    self.src_pool.update(json.load(f))
                with open(os.path.join(ds_path, f"{ds}.json")) as f:
                    self.label_ratios.update(json.load(f))
                with open(os.path.join(ds_path, f"color_map.json")) as f:
                    self.label_colors[ds] = {k: v for k, v in json.load(f).items() if k not in self.INVALID_CLS}
                if os.path.isfile(os.path.join(ds_path, "meta_info.json")):
                    with open(os.path.join(ds_path, "meta_info.json")) as f:
                        self.meta_info[ds] = json.load(f)
                if isinstance(fg_ratio, float):
                    self.fg_ratio[ds] = fg_ratio
                else:
                    for k, v in fg_ratio.items():
                        if k in ds_path:
                            self.fg_ratio[ds] = v
                            continue
                self.dataset_full_paths[ds] = ds_path

        print(f"Datasets loaded for mode {mode}: {self.dataset_full_paths.keys()}")
        self._parse_samples()

        self.uni_transform = uni_transform
        self.aug_transform = aug_transform
        if self.uni_transform is None:
            self.uni_transform = MapTransformTrain(image_size=self.img_size) if 'train' in mode else MapTransformTest(image_size=self.img_size)
        
        assert len(self.file_list) > 0, f"No samples found in info files {info_file_dict} for mode {mode} with filtering keys {self.filtering_keys}."
        print(f"Number of {mode} samples: {len(self.file_list)}")
        print(f"Number of non-repetative {mode} images: {len(set([x.split('/')[1] for x in self.file_list]))}")
    
    def _parse_samples(self):
        self.file_list = []
        pos_file_list = []
        neg_file_list = []
        removed_count = 0

        for tgt_sample, pairing_dict in self.src_pool.items():
            for cls_name, pairing_info in pairing_dict.items():
                if cls_name in self.meta_info.get(tgt_sample.split('.', 1)[0], {}).get('ignore_cls', []):
                    continue
                if len(pairing_info['src_samples']) == 0:
                    removed_count += 1
                    continue
                
                cls_full_name = f"{cls_name}.{tgt_sample.split('.', 1)[0]}"
                if self.mode == "train":
                    if self.label_ratios[tgt_sample].get(cls_name, 0.0) > self.fg_ratio[tgt_sample.rsplit(".", 1)[0]]:
                        pos_file_list.append(f"{cls_full_name}.pos/{tgt_sample}")
                    else:
                        neg_file_list.append(f"{cls_full_name}.neg/{tgt_sample}")
                elif self.mode == "test":
                    if tgt_sample in self.label_ratios:
                        if self.label_ratios[tgt_sample].get(cls_name, 0.0) > self.fg_ratio[tgt_sample.rsplit(".", 1)[0]]:
                            pos_file_list.append(f"{cls_full_name}.pos/{tgt_sample}")
                        else:
                            neg_file_list.append(f"{cls_full_name}.neg/{tgt_sample}")
                    else:
                        removed_count += 1
                else:
                    # use all samples
                    pos_file_list.append(f"{cls_full_name}.pos/{tgt_sample}")

        if removed_count > 0:
            print(f"Removed {removed_count} samples with no source examples.")

        self.file_list = pos_file_list + neg_file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def _build_candidate_pools(self):
        """Union of all retrieved source samples per (subdataset, class).

        This is the set of annotated tiles a user would have to provide for a subset, i.e. the
        annotation budget. Drawing from it uniformly is the control condition for the retrieval
        experiment: same budget, same candidates, similarity ranking removed.
        """
        pools = {}
        for tgt_full_name, pairing_dict in self.src_pool.items():
            ds = tgt_full_name.rsplit(".", 1)[0]
            for cls_name, pairing_info in pairing_dict.items():
                pools.setdefault((ds, cls_name), set()).update(pairing_info['src_samples'])
        self.candidate_pools = {k: sorted(v) for k, v in pools.items()}

    def _select_src_samples(self, cls_name, tgt_full_name):
        """Pick ``num_src`` source samples for one target image and class.

        ``src_selection`` controls how:
            "retrieved" - uniformly among the retrieved neighbours of this target (default,
                          the behaviour of every run so far)
            "topk"      - the ``num_src`` closest neighbours, in ranking order
            "random"    - uniformly from all annotated tiles of the same subset and class,
                          ignoring similarity (the control for the retrieval claim)
        """
        cls_key = cls_name.split('.')[0]
        src_samples = self.src_pool[tgt_full_name][cls_key]['src_samples']
        assert len(src_samples) > 0, \
            f"No source samples found for target image {tgt_full_name} and class {cls_name}."

        if self.src_selection == "topk":
            # the pairing files store the neighbours in increasing distance
            src_full_names = src_samples[:self.num_src]
            while len(src_full_names) < self.num_src:
                src_full_names.append(src_samples[len(src_full_names) % len(src_samples)])
            return list(src_full_names)

        if self.src_selection == "random":
            if not hasattr(self, "candidate_pools"):
                self._build_candidate_pools()
            ds = tgt_full_name.rsplit(".", 1)[0]
            src_samples = self.candidate_pools.get((ds, cls_key), src_samples)

        return np.random.choice(
            src_samples,
            self.num_src,
            replace=len(src_samples) < self.num_src
        ).tolist()
    
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
    
    def _select_notions(self, tgt_full_name, cls_full_name, dataset, src_full_names):
        cls_name = cls_full_name.split(".")[0]
        # if cls_name == "negative":
        #     return ["negative"], [self.label_colors[dataset]["negative"]], [0]
        all_cls_names = list(self.label_colors[dataset].keys())
        if self.num_notions == 1:
            selected_classes = [cls_name]
            selected_colors = [self.label_colors[dataset][cls_name]]
            selected_cls_ids = [all_cls_names.index(cls_name) + 1]
            return selected_classes, selected_colors, selected_cls_ids
        else:
            # in addition to the main class, randomly select other classes from the target image to improve diversity
            tgt_cls_ratios = self.label_ratios[tgt_full_name]
            src_cls_ratios = {k: 0.0 for k in self.label_colors[dataset].keys()}
            for src_full_name in src_full_names:
                for cls, ratio in self.label_ratios[src_full_name].items():
                    if ratio == 0.0 or cls in self.INVALID_CLS:
                        continue
                    src_cls_ratios[cls] += ratio
            # pos_cls: classes present in both source and target images
            pos_cls = []
            # neg_cls: classes not present in the target image
            neg_cls = []
            for cls in tgt_cls_ratios.keys():
                if cls == cls_name:
                    continue
                if tgt_cls_ratios[cls] > self.fg_ratio[dataset] and src_cls_ratios.get(cls, 0.0) > self.fg_ratio[dataset]:
                    pos_cls.append(cls)
                else:
                    neg_cls.append(cls)
            # remove invalid classes
            pos_cls = [c for c in pos_cls if c not in self.meta_info.get(dataset, {}).get('ignore_cls', [])]
            neg_cls = [c for c in neg_cls if c not in self.meta_info.get(dataset, {}).get('ignore_cls', [])]
            # select pos cls with probability self.pos_prob and neg cls with probability 1 - self.pos_prob
            p = 1 - self.pos_prob if cls_full_name.endswith(".neg") else self.pos_prob
            
            neg_cls = [c for c in neg_cls if c not in self.INVALID_CLS]
            pos_neg = pos_cls + neg_cls
            pos_neg_probs = [p / max(1, len(pos_cls)),] * len(pos_cls) + \
                [(1 - p) / max(1, len(neg_cls)),] * len(neg_cls)
            selected_classes = [cls_name]
            if len(pos_neg) > 0:
                selected_classes += list(np.random.choice(
                    pos_neg, 
                    self.num_notions - 1, 
                    replace=len(pos_neg)<(self.num_notions - 1), 
                    p=np.array(pos_neg_probs) / np.sum(pos_neg_probs)
                ))

            selected_cls_ids = [all_cls_names.index(cls_name) + 1 for cls_name in selected_classes]
            selected_colors = [self.label_colors[dataset].get(c, None) for c in selected_classes]
        return selected_classes, selected_colors, selected_cls_ids
    
    def _construct_multi_notion_label_via_colors(self, lbl, selected_colors):
        palette = lbl.getpalette()
        palette = np.array(palette).reshape(-1, 3).tolist()
        selected_color_inds = [palette.index(color) if color is not None else None for color in selected_colors]
        assert 0 < len(selected_color_inds) <= self.num_notions, \
            f"Number of selected notions {len(selected_color_inds)} should be > 0 and < {self.num_notions}."
        multi_notion_lbl = np.zeros((lbl.size[1], lbl.size[0], self.num_notions,), dtype=np.uint8)
        lbl = np.array(lbl.convert("P"))
        for i, color_idx in enumerate(selected_color_inds):
            if color_idx is not None:
                multi_notion_lbl[..., i][lbl == color_idx] = 1
            # add the ignore mask
            multi_notion_lbl[..., i][lbl == 255] = 255
        return multi_notion_lbl
    
    def _constrct_multi_notion_label_via_classes(self, lbl, selected_cls_ids):
        """Faster than via colors."""
        multi_notion_lbl = np.zeros((lbl.size[1], lbl.size[0], self.num_notions,), dtype=np.uint8)
        lbl = np.array(lbl.convert("P"))
        for i, cls_idx in enumerate(selected_cls_ids):
            if cls_idx is not None:
                multi_notion_lbl[..., i][lbl == cls_idx] = 1
            # add the ignore mask
            multi_notion_lbl[..., i][lbl == 255] = 255
        return multi_notion_lbl

    
    def _load_one_img_lbl_pair(self, img_full_name, selected_cls_ids):
        img, lbl = self._load_image_label(img_full_name)
        multi_notion_lbl = self._constrct_multi_notion_label_via_classes(lbl, selected_cls_ids)
        img, multi_notion_lbl = self.uni_transform(img, multi_notion_lbl)
        return img, multi_notion_lbl
    
    
    def __getitem__(self, index):
        cls_name, tgt_full_name = self.file_list[index].split("/")
        dataset, img_name = tgt_full_name.rsplit(".", 1)

        src_full_names = self._select_src_samples(cls_name, tgt_full_name)

        selected_classes, selected_colors, selected_cls_ids = self._select_notions(tgt_full_name, cls_name, dataset, src_full_names)               

        # load images and labels
        imgs = []
        lbls = []
        full_names = [tgt_full_name] + src_full_names
        for img_full_name in full_names:
            img, multi_notion_lbl = self._load_one_img_lbl_pair(img_full_name, selected_cls_ids)
            imgs.append(img)
            lbls.append(multi_notion_lbl)
        if self.uni_transform is not None:
            imgs, lbls = self.uni_transform(imgs, lbls)
        if self.aug_transform is not None:
            imgs, lbls = self.aug_transform(imgs, lbls)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)[..., ::4, ::4]
        tgt_lbl = lbls[0]
        full_cls_names = [f"{dataset}/{x}" for x in selected_classes] 
        aux = {'src_name': src_full_names, 'tgt_name': tgt_full_name, 'cls_name': full_cls_names, 'cls_color': selected_colors}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux
    
    def vis_data(self, imgs, lbls, selected_classes):
        import matplotlib.pyplot as plt
        num_imgs = len(imgs)
        fig, axs = plt.subplots(2, num_imgs, figsize=(4*num_imgs, 8))
        for i in range(num_imgs):
            axs[0, i].imshow(imgs[i])
            axs[0, i].set_title(f"Image {i}")
            axs[0, i].axis('off')
            lbl = lbls[i]
            lbl[lbl == 1] = 128
            axs[1, i].imshow(lbl[..., 0].astype(float) / 255, cmap='jet', vmin=0, vmax=1)
            axs[1, i].set_title(f"Label {i}: {selected_classes}")
            axs[1, i].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
            cls_full_name = fname.split('/')[0]
            # loc = '.'.join(cls_full_name.split('.')[:2])  
            loc = cls_full_name
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



