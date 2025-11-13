# This dataset is downloaded from scene flow data from kitti: http://www.cvlibs.net/datasets/kitti

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from .syntra_data_converter import SyntraDataConverter
from .cityscapes_labels import labels, get_color_map_for_eval_classes # kitti uses the same labels as cityscapes

color_map = get_color_map_for_eval_classes()


class KittiConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=512):
        # use the scene flow data folder as input
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size)
    
    def _parse_split_img_lbl(self, split, skip_unlabeled=True):
        assert split in ['train', 'test']
        img_files = glob(os.path.join(self.customized_data_path, f'{split}ing/image_2', "*.png"))

        samples = []
        # parse img and label pairs
        for img_file in img_files:
            base_name = os.path.basename(img_file)
            lbl_file = os.path.join(self.customized_data_path, f'{split}ing/semantic', base_name)
            if not os.path.exists(lbl_file):
                print(f"Label file {lbl_file} does not exist for image {img_file}, skip.")
                lbl_file = None
                if skip_unlabeled:
                    continue
            samples.append( (img_file, lbl_file))
        return samples
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).replace('.png', '')
        return base_name
    
    def _get_id_map(self):
        return {label.id: label.trainId if label.trainId == 255 else label.trainId + 1 for label in labels if label.id != -1}
    





    



