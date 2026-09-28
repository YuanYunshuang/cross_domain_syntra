from glob import glob
import os

from PIL import Image
from tifffile import imread
import torch
import numpy as np
import json
from typing import Dict, Optional

from torch.utils.data import Dataset
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from typing import List
from .trainsform import MapTransformTrain, MapTransformTest




class SMOLDataset(Dataset):
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
                 source_file: Optional[str] = None,
                 target_file: Optional[str] = None,
                 src_fg_max: Optional[float] = 0.9,
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
        """
        self.data_root = data_root
        self.num_notions = num_notions
        self.num_src = num_src
        self.mode = mode
        self.fg_ratio = {}
        self.pos_prob = pos_prob
        self.img_size = img_size

        # load samples
        self.file_list = []
        self.src_pool = {}
        self.label_ratios = {}
        self.label_colors = {}
        self.meta_info = {}
        self.dataset_full_paths = {}
        self.filtering_keys = [x.strip() for x in filtering_keys.split(",")] if filtering_keys is not None else None
        self.source_file = source_file
        self.target_file = target_file
        # Upper bound on a source prompt's class coverage. SMOL-MapSeg hard-coded 0.9, and that
        # stays the default so every archived SMOL run reproduces. It used to be a literal in
        # `_parse_source_samples`, so `--src-fg-max` reached this class through **kwargs and was
        # silently dropped. None, or any value above 1 (`--src-fg-max inf`), means no cap, which
        # matches cd_syntra2's default.
        self.src_fg_max = src_fg_max

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
                if not os.path.isfile(os.path.join(ds_path, info_file)):
                    Warning(f"Info file {info_file} does not exist in dataset folder {ds_path}. Skip this dataset.")
                    continue
                
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
                self._parse_source_target_samples(info_file, ds_path, ds)
                self.dataset_full_paths[ds] = ds_path

        print(f"Datasets loaded for mode {mode}: {self.dataset_full_paths.keys()}")
        self._parse_source_samples()
        if self.mode != "train":
            self._parse_classes()

        self.uni_transform = uni_transform
        self.aug_transform = aug_transform
        if self.uni_transform is None:
            self.uni_transform = MapTransformTrain(image_size=self.img_size) if 'train' in mode else MapTransformTest(image_size=self.img_size)
        
        assert len(self.file_list) > 0, f"No samples found in info files {info_file_dict} for mode {mode} with filtering keys {self.filtering_keys}."
        print(f"Number of {mode} samples: {len(self.file_list)}")

    def _parse_classes(self):
        file_list = []
        for full_name in self.file_list:
            ds, name = full_name.rsplit('.', 1)
            availabel_classes = [k for k in self.label_ratios[full_name].keys() if k not in self.INVALID_CLS]
            for cls in availabel_classes:
                available_src = self.loading_list[ds][cls]
                if len(available_src) > 0 and cls not in self.INVALID_CLS:
                    file_list.append((full_name, cls))
        self.file_list = file_list

    def _parse_source_target_samples(self, info_file: str, ds_path: str, ds: str):
        def _load_samples(file_path: str) -> List[str]:
            if file_path.endswith('.json'):
                with open(file_path) as f:
                    samples = []
                    for s in json.load(f).values():
                        samples.extend(s)
                    samples = list(set(samples))
            else:
                with open(file_path) as f:
                    samples = [line.strip() for line in f.readlines()]
            return samples
        # load source and target samples
        #
        # Unlike cd_syntra2, this reads the *sample lists* rather than the pairing JSON - SMOL
        # has no retrieval, so it needs the pool a prompt may be drawn from, not a ranked
        # neighbour list. Which lists those are used to be inferred from the pairing's filename,
        # which only ever recognised `cls_selection_<n>` and `train_samples` and fell through to
        # the *whole training split* for anything else. That fallback was silent: pointing this
        # at a budgeted selection reported the budget in the config and trained on all 14,675
        # tiles. Explicit `source_file` / `target_file` now take precedence, and the inference
        # raises rather than guessing when it does not recognise the name.
        if self.source_file:
            source_file = self.source_file
            if self.mode == 'train':
                # Only training needs a target list named; test and val targets are always the
                # whole split, which is the point - every arm is scored on identical data.
                if not self.target_file:
                    raise ValueError(
                        "source_file was given without target_file in train mode. The targets "
                        "would silently fall back to the full split, which is the bug this "
                        "check exists to prevent.")
                target_file = self.target_file
            else:
                # The whole split by default, so every arm is scored on identical data. An
                # explicit target list overrides it, which is how a filtered test set (e.g. one
                # with the tiles overlapping training removed) is scored identically across
                # SMOL and cd_syntra2.
                target_file = self.target_file or f"{self.mode}_samples.txt"
        elif 'sample_pair' in info_file and info_file.endswith('_intracls.json'):
            # Only the legacy `_intracls` naming can be inferred, and only because the archive's
            # runs depend on it. The suffix has to be checked, not just stripped: on a `_gap0`
            # name the strip is a no-op, `[-2]` is then '100' rather than 'selection', and the
            # target silently became the whole 14,675-tile split while the source was read
            # correctly from the selection - a half-right load, which is worse than a failure.
            info = info_file.replace('sample_pair_', '').replace('_intracls.json', '')
            head = info.split('_')[0]
            if head == 'cls':
                source_file = f"cls_selection_{info.split('_')[2]}.json"
            elif head == 'train':
                source_file = "train_samples.txt"
            else:
                raise ValueError(
                    f"SMOLDataset cannot infer the source list from '{info_file}'. It only "
                    f"recognises 'cls_selection_<n>' and 'train_samples'. Pass source_file= and "
                    f"target_file= explicitly.")
            if info.split('_')[-2] == 'selection' and self.mode == 'train':
                target_file = f"cls_selection_{info.split('_')[-1]}.json"
            else:
                target_file = f"{self.mode}_samples.txt"

        elif 'sample_pair' in info_file:
            raise ValueError(
                f"SMOLDataset cannot infer its sample lists from '{info_file}'. Inference is "
                f"only supported for the legacy '_intracls' naming; anything else - including "
                f"every '_gap0' pairing this project now builds - must name its lists with "
                f"source_file= and target_file=, or the target silently becomes the full "
                f"training split.")

        else:
            info = info_file.split('_')
            source_file = f"{info[2]}_samples.txt"
            target_file = f"{info[4]}_samples.txt"

        print(f"source: {source_file}, target: {target_file}, dataset: {ds}.")
        self.src_pool[ds] = _load_samples(os.path.join(ds_path, source_file))
        self.file_list += _load_samples(os.path.join(ds_path, target_file))
                
        # self.file_list = ['railway.111_sie'] # for debug


    def _parse_source_samples(self):
        loading_list_src = {}
        for full_name_list in self.src_pool.values():
            for full_name in full_name_list:
                ds, name = full_name.rsplit('.', 1) # ds: subdataset name, key: image name
                info = self.label_ratios[full_name]
        
                if ds not in loading_list_src:
                    loading_list_src[ds] = {c: [] for c in info.keys() if c not in self.INVALID_CLS}
                for k, v in info.items(): # k: class, v: ratio
                    if k in self.INVALID_CLS:
                        continue
                    if self.fg_ratio[ds] < v and (self.src_fg_max is None or v < self.src_fg_max):
                        loading_list_src[ds][k].append(full_name)
        self.loading_list = loading_list_src

    def select_tgt_cls(self, tgt_full_name):
        fg_ratio = self.fg_ratio[tgt_full_name.rsplit(".", 1)[0]]

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
        if self.mode == "train":
            tgt_full_name = self.file_list[index]
            selected_cls = self.select_tgt_cls(tgt_full_name)    
        else:
            tgt_full_name, selected_cls = self.file_list[index]

        dataset, _ = tgt_full_name.rsplit(".", 1)
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
            img = Image.open(os.path.join(self.dataset_full_paths[ds], 'imgs', img_name + '.png'))
            img = np.array(img)
            lbl = Image.open(os.path.join(self.dataset_full_paths[ds], 'lbls', img_name + '.png'))
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
        imgs, lbls = self.uni_transform(imgs, lbls)
        # self.visualize(src_img, src_lbl, tgt_img, tgt_lbl)
        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.stack(lbls[1:], 0)
        tgt_lbl = lbls[0]

        aux = {'src_name': src_full_names, 
               'tgt_name': tgt_full_name,
               'cls_name': [f"{dataset}/{selected_cls}"],
               'cls_color': selected_colors,
               'cls_id': [[i+1 for i in range(len(selected_color_inds))]]}

        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux

    def subdataset_id(self, index: int) -> int:
        # Derive subdataset id from filename prefix before the first '.'
        # Example: "locA.some_image_name_001.png" -> "locA"
        def get_loc(fname):
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
    
    def prepare_inputs(self, src_imgs, src_lbls, tgt_img, tgt_lbl, aux):
        """
        Prepare inputs for the model by constructing multi-notion labels on the GPU.
        
        Arguments:
            src_imgs (torch.Tensor): Source images tensor of shape (B, T-1, 3, H, W).
            src_lbls (torch.Tensor): Source labels tensor of shape (B, T-1, H/4, W/4).
            tgt_img (torch.Tensor): Target image tensor of shape (B, 3, H, W).
            tgt_lbl (torch.Tensor): Target label tensor of shape (B, H, W).
            aux (dict): Auxiliary information containing class names and other metadata.
        """
        # construct multi-notion labels on the GPU
        B, T_1, C, H, W = src_imgs.shape

        if self.aug_transform is not None:
            imgs = torch.cat([tgt_img.unsqueeze(1), src_imgs], dim=1)  # (B, T, C, H, W)
            lbls = torch.cat([tgt_lbl.unsqueeze(1), src_lbls], dim=1)  # (B, T, H, W)
            imgs, lbls = self.aug_transform(imgs, lbls)
            imgs = imgs.reshape(B, T_1 + 1, C, H, W)
            lbls = lbls.reshape(B, T_1 + 1, H, W)
            tgt_img = imgs[:, 0]
            src_imgs = imgs[:, 1:]
            tgt_lbl = lbls[:, 0]
            src_lbls = lbls[:, 1:, ::4, ::4]
        else:
            src_lbls = src_lbls.reshape(B, T_1, H, W)
            src_lbls = src_lbls[..., ::4, ::4]
            tgt_lbl = tgt_lbl.reshape(B, H, W)

        selected_cls_ids = [x['cls_id'] + [-1] * (self.num_notions - len(x['cls_id'])) for x in aux]
        selected_cls_ids = torch.tensor(selected_cls_ids, device=src_imgs.device, dtype=torch.int16)  # (B, num_notions)

        multi_notions_src_lbls = src_lbls.unsqueeze(2).repeat(1, 1, self.num_notions, 1, 1)
        masks_pos = multi_notions_src_lbls.to(torch.int16) == selected_cls_ids.reshape(B, 1, self.num_notions, 1, 1)
        masks_neg = torch.logical_not(masks_pos) & (multi_notions_src_lbls != 255)
        multi_notions_src_lbls[masks_pos] = 1
        multi_notions_src_lbls[masks_neg] = 0
        multi_notions_tgt_lbls = tgt_lbl.unsqueeze(1).repeat(1, self.num_notions, 1, 1)
        masks_pos = multi_notions_tgt_lbls.to(torch.int16) == selected_cls_ids.reshape(B, self.num_notions, 1, 1)
        masks_neg = torch.logical_not(masks_pos) & (multi_notions_tgt_lbls != 255)
        multi_notions_tgt_lbls[masks_pos] = 1
        multi_notions_tgt_lbls[masks_neg] = 0

        return src_imgs, multi_notions_src_lbls, tgt_img, multi_notions_tgt_lbls, aux
    




if __name__ == '__main__':
    dataset = SMOLDataset("/koko/datasets/SMOL_syntra", info_file="train_100shot.txt", mode="train")
    # dataset = SynTraDataset("/koko/datasets/SMOL_syntra", info_file="pairing_train_100shot_train_100shot_intercls.json", mode="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    for d in dataset:
         src_img, src_lbl, tgt_img, tgt_lbl, aux = d
         if src_lbl.sum() == 0 and tgt_lbl.sum() > 0:
             print(src_img.shape, src_lbl.shape, tgt_img.shape, tgt_lbl.shape, aux)
             print("empty src lbl")
