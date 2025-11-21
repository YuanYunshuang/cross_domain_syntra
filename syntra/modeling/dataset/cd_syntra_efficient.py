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
from .cd_syntra import CrossDomainSynTraDataset
from typing import Dict, Optional


class CrossDomainSynTraDatasetE(CrossDomainSynTraDataset):
    """
    Inherits from CrossDomainSynTraDataset and overrides methods for optimized performance.
    Changes: 
    - labels will kept originally as loaded (uint8), the construction of multi-notion labels should 
        be done in the Model Class and on the GPU.
    - added lazy caching for loading data. The images and lbls would be cached after the first load to speed up.
    """
    INVALID_CLS = ['background', 'nonlabeled']
    def __init__(self, 
                 cache_data: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.cache_data = cache_data
        self.cache = {}
    
    
    def __getitem__(self, index):
        cls_name, tgt_full_name = self.file_list[index].split("/")
        dataset, img_name = tgt_full_name.rsplit(".", 1)

        src_full_names = self._select_src_samples(cls_name, tgt_full_name)

        selected_classes, selected_colors, selected_cls_ids = self._select_notions(cls_name, dataset, src_full_names)               

        # load images and labels
        imgs = []
        lbls = []
        full_names = [tgt_full_name] + src_full_names
        for img_full_name in full_names:
            if img_full_name in self.cache:
                img, lbl = self.cache[img_full_name]
            else:
                 img, lbl = self._load_image_label(img_full_name)
                 if self.uni_transform is not None:
                    img, lbl = self.uni_transform(img, lbl)
                 if self.cache_data:
                     self.cache[img_full_name] = (img, lbl)

            imgs.append(img)
            lbls.append(lbl)

        src_imgs = torch.stack(imgs[1:], 0)
        tgt_img = imgs[0]
        src_lbls = torch.cat(lbls[1:], 0)
        tgt_lbl = lbls[0].squeeze(0)
        full_cls_names = [f"{dataset}/{x}" for x in selected_classes] 
        aux = {'src_name': src_full_names, 
               'tgt_name': tgt_full_name, 
               'cls_name': full_cls_names, 
               'cls_color': selected_colors,
               'cls_id': selected_cls_ids}
        # print(src_img_name, tgt_image_name)
        return src_imgs, src_lbls, tgt_img, tgt_lbl, aux
    

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
            imgs, lbls = self.aug_transform(imgs.flatten(0, 1), lbls.flatten(0, 1))
            imgs = imgs.reshape(B, T_1 + 1, C, H, W)
            lbls = lbls.reshape(B, T_1 + 1, H, W)
            tgt_img = imgs[:, 0]
            src_imgs = imgs[:, 1:]
            tgt_lbl = lbls[:, 0]
            src_lbls = lbls[:, 1:, ::4, ::4]
        else:
            src_lbls = src_lbls[:, :, ::4, ::4]

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
        

    
    


