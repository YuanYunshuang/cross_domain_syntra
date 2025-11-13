# This dataset is downloaded from https://zenodo.org/records/13126114

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from .syntra_data_converter import SyntraDataConverter
from .cityscapes_labels import labels, get_color_map_for_eval_classes

color_map = get_color_map_for_eval_classes()


class CityScapesConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=512):
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size)
    
    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'val', 'test']
        img_files = glob(os.path.join(self.customized_data_path, 'leftImg8bit', split, "**/*.png"))

        samples = []
        # parse img and label pairs
        for img_file in img_files:
            base_name = os.path.basename(img_file).replace('_leftImg8bit.png', '')
            lbl_file = glob(os.path.join(self.customized_data_path, f"**/{base_name}_gtFine_labelIds.png"), recursive=True)
            assert len(lbl_file) == 1, f"Expected one label file for image {img_file}, but found {len(lbl_file)}."
            lbl_file = lbl_file[0]

            samples.append( (img_file, lbl_file))
        return samples
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).replace('_leftImg8bit.png', '')
        return base_name
    
    def _get_id_map(self):
        return {label.id: label.trainId if label.trainId == 255 else label.trainId + 1 for label in labels if label.id != -1}
    





    



