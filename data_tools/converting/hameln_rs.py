# This dataset is downloaded from scene flow data from kitti: http://www.cvlibs.net/datasets/kitti

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from tifffile import imread

from .syntra_data_converter import SyntraDataConverter

color_map = {
  "building": [180, 180, 180],           #1 neutral gray
  "coniferous_trees": [15, 100, 40],     #2 dark forest green
  "crop": [255, 200, 40],                #3 bright yellow-gold
  "deciduous_trees": [60, 180, 75],      #4 light fresh green
  "grass": [120, 220, 60],               #5 bright lime green
  "grove": [80, 130, 60],                #6 olive green
  "loose_material": [210, 180, 120],     #7 sandy beige
  "reed": [160, 90, 160],                #8 muted purple
  "running_water": [40, 90, 200],        #9 clear blue
  "shrubbery": [100, 160, 100],          #10 medium sage green
  "standing water": [20, 50, 140],       #11 deep navy blue
  "superstructure_work": [150, 50, 30],  #12 reddish brown
  "underground_working": [120, 90, 60],  #13 earthy brown
}


class HamelnRSConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, manual=True, fg_ratio=0.001, img_size=384, **kwargs):
        """
        Data structure in customized_data_path:

        """
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        if manual:
            self.customized_data_path = os.path.join(customized_data_path, 'Hameln_Manual_v3') 
        else:
            self.customized_data_path = os.path.join(customized_data_path, 'Hameln_ATKIS_v3')
        with open(os.path.join(customized_data_path, 'train_val_test_indices.json'), 'r') as f:
            self.dataset_split = json.load(f)

    
    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test', 'val']
        indices = self.dataset_split[split]
        samples = []
        for epoch in os.listdir(self.customized_data_path):
            epoch_path = os.path.join(self.customized_data_path, epoch)
            for sample_id in indices:
                sample_path = os.path.join(epoch_path, str(sample_id))

                img_file = os.path.join(sample_path, 'R_G_B.tif')
                lbl_file = os.path.join(sample_path, 'L.tif')

                # if not '2000/40' in img_file:
                #     continue
                samples.append((img_file, lbl_file))
        return samples
    
    def _load_original_lbl(self, lbl_file):
        """Load tif image file."""

        lbl = imread(lbl_file)
        assert lbl.max() <= len(color_map), f"Label image has value {lbl.max()} which exceeds number of classes {len(color_map)}."
        # assert lbl.dtype == np.uint8, f"Label dtype must be uint8, got {lbl.dtype}."
        # assert lbl.ndim == 2, f"Label image must be 2D, got shape {lbl.shape}."
        lbl += 1  # shift to 0~N
        lbl[lbl == 0] = 255  # ignore index
        lbl = lbl.astype(np.uint8)
        lbl = Image.fromarray(lbl)

        return lbl
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file)
        epoch, sample_id = img_file.split('/')[-3:-1]
        base_name = f"{epoch}_{sample_id}"
        base_name = base_name.replace('.tif', '')
        return base_name
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    
    # def _construct_color_palette(self):
    #     """
    #     Constructs a color palette list from the color map.
    #     Returns:
    #         list: A flat list representing the color palette.
    #     """
    #     palette = [0, 0, 0]  # background color
    #     for cls_name in self.color_map.keys():
    #         palette.extend(self.color_map[cls_name])
    #     # pad the palette to have 256 colors
    #     palette.extend([0, 0, 0] * (256 - len(self.color_map) - 2))
    #     # set the last color to white as nonlabled
    #     palette.extend([255, 255, 255])
    #     self.palette = palette
    

    





    



