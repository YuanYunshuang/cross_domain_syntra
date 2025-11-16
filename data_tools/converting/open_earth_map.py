# This dataset is downloaded from scene flow data from kitti: http://www.cvlibs.net/datasets/kitti

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from .syntra_data_converter import SyntraDataConverter

color_map = { # https://open-earth-map.org/overview_oem.html
  "Bareland": [128, 0, 0],
  "Rangeland": [0, 255, 36],
  "Developed space": [148, 148, 148],
  "Road": [255, 255, 255],
  "Tree": [34, 97, 38],
  "Water": [0, 69, 255],
  "Agriculture land": [75, 181, 73],
  "Building": [222, 31, 7]
}


class OEMonverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=512):
        # use the scene flow data folder as input
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size)
    
    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test', 'val']
        skip_unlabeled=False if split=='test' else True
        with open(os.path.join(self.customized_data_path, f'{split}.txt'), 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

        samples = []
        # parse img and label pairs
        for filename in filenames:
            city = filename.split('_')[0]
            img_file = os.path.join(self.customized_data_path, city, "images", filename, filename)
            if os.path.exists(img_file) is False:
                print(f"Image file {img_file} does not exist, skip.")
                continue
            lbl_file = os.path.join(self.customized_data_path, city, "labels", filename, filename)
            if not os.path.exists(lbl_file):
                print(f"Label file does not exist for image {img_file}, skip lbl.")
                if skip_unlabeled:
                    continue
                else:
                    lbl_file = None
            samples.append((img_file, lbl_file))
        return samples
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).replace('.tif', '')
        return base_name
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    





    



