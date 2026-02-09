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
  "forest": [34, 139, 34],          
  "grass": [124, 252, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
}

color_map_zs1 = { 
  "settlement": [220, 20, 60],               
}

color_map_zs2 = {
  "sport_area": [255, 140, 0],        
  "industrial_area": [255, 215, 0],   
  "federal_highway": [138, 43, 226],
  "state_road": [75, 0, 130],
}


class HismapZSConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, filtering_keys=None,
                 zeroshot=True, fg_ratio=0.01, img_size=384, **kwargs):
        """
        Data structure in customized_data_path:

        """
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        self.filtering_keys = filtering_keys
        self.zeroshot = zeroshot
        self.dataset_split = {}
        if zeroshot:
            self.customized_data_path = os.path.join(customized_data_path, 'hameln2') 
            with open(os.path.join(customized_data_path, 'train_hameln2.txt'), 'r') as f:
                self.dataset_split['train'] = f.read().splitlines()
            with open(os.path.join(customized_data_path, 'test_hameln2.txt'), 'r') as f:
                self.dataset_split['test'] = f.read().splitlines()
        else:
            raise NotImplementedError("Only zeroshot=True is supported for HiamapZSConverter.")

    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test']
        split_samples = self.dataset_split[split]
        samples = []
        for sample in split_samples:
            s = sample.split('.')[1]
            if self.filtering_keys is not None and all([fk not in s for fk in self.filtering_keys]):
                continue
            img_file = os.path.join(self.customized_data_path, 'imgs', f'{s}.png')
            lbl_file = os.path.join(self.customized_data_path, 'lbls', f'{s}.png')
            samples.append((img_file, lbl_file))
        return samples
    
    def _load_original_lbl(self, lbl_file):
        """Load tif image file."""
        lbl = Image.open(lbl_file)
        if self.zeroshot:
            if "settlement" in self.color_map:
                return lbl
            else:
                lbl = np.array(lbl)
                lbl[lbl>0] -= 1  # no settlement class
                lbl = lbl.astype(np.uint8)
                lbl = Image.fromarray(lbl)

        return lbl
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file)
        if self.zeroshot:
            base_name = base_name.split('.')[0].replace('c3821_', '')
        else:
            raise NotImplementedError("Only zeroshot=True is supported for HiamapZSConverter.")     
        return base_name    
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    

    





    



