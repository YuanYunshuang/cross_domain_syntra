# This dataset is downloaded from scene flow data from kitti: http://www.cvlibs.net/datasets/kitti

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from tifffile import imread

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
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=384, **kwargs):
        """
        Data structure in customized_data_path:
        ├── OpenEarthMap_wo_xBD
        │   ├── aachen
        │   │   ├── images
        │   │   └── labels
        │   ├── ...
        └── xview2
            ├── hold
            │   ├── images
            │   └── labels
            ├── test
            │   ├── images
            │   └── labels
            ├── tier1
            │   ├── images
            │   └── labels
            └── tier3
                ├── images
                └── labels

        """
        # use the scene flow data folder as input
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        self.oem_path = os.path.join(self.customized_data_path, 'OpenEarthMap_wo_xBD')
        self.xbd_path = os.path.join(self.customized_data_path, 'xview2')
        self.oem_to_xbd_map = {}
        self.xbd_to_oem_map = {}
        with open(os.path.join(self.oem_path, 'xbd_files.csv'), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                oem_filename = items[1]
                xbd_filename = items[0].split('.')[0] + '.tif'
                self.oem_to_xbd_map[oem_filename] = xbd_filename
                self.xbd_to_oem_map[xbd_filename] = oem_filename
    
    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test', 'val']
        skip_unlabeled=False if split=='test' else True
        with open(os.path.join(self.oem_path, f'{split}.txt'), 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

        samples = []
        # parse img and label pairs
        for filename in filenames:
            city = filename.rsplit('_', 1)[0]
            img_file = os.path.join(self.oem_path, city, "images", filename)
            if not os.path.exists(img_file):
                xbd_filename = self.oem_to_xbd_map.get(filename, None)
                for subdir in ['hold', 'test', 'tier1', 'tier3']:
                    img_file = os.path.join(self.xbd_path, subdir, 'images', xbd_filename)
                    if os.path.exists(img_file):
                        break
            if not os.path.exists(img_file):
                print(f"Image file {img_file} does not exist, skip.")
                continue
            lbl_file = os.path.join(self.oem_path, city, "labels", filename)
            if not os.path.exists(lbl_file):
                print(f"Label file does not exist for image {img_file}, skip lbl.")
                if skip_unlabeled:
                    continue
                else:
                    lbl_file = None
            samples.append((img_file, lbl_file))
        return samples
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file)
        if self.xbd_path in img_file:
            base_name = self.xbd_to_oem_map.get(base_name, None)
            assert base_name is not None, f"Cannot find corresponding OEM filename for {img_file}"
        base_name = base_name.replace('.tif', '')
        return base_name
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    





    



