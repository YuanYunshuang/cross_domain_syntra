# This dataset is downloaded from https://zenodo.org/records/13126114

import json
from glob import glob
import os
from PIL import Image
import numpy as np

from .syntra_data_converter import SyntraDataConverter

color_map = {
  "ABBA": [132, 74, 201],
  "ACPE": [14, 191, 102],
  "ACRU": [222, 57, 63],
  "ACSA": [88, 208, 234],
  "Acer": [247, 158, 31],
  "BEAL": [69, 132, 240],
  "BEPA": [185, 92, 38],
  "FAGR": [41, 173, 19],
  "LALA": [236, 57, 174],
  "Mort": [123, 123, 123],
  "Picea": [21, 96, 150],
  "PIST": [50, 150, 50],
  "Populus": [255, 255, 0],
  "THOC": [210, 105, 30],
  "TSCA": [128, 0, 128],
}




class ForestMonitoringTreesConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=384, month='september', **kwargs):
        """
        Note: This dataset is downloaded from https://zenodo.org/records/13126114. The label has hierachical structure, 
        but we only use the species-level (the first channel of the label images) labels for conversion. Besides, we only
        use the data from september for segmentation.

        Data structure in customized_data_path:
        ├── classes_onlymyriam_augmented.json
        ├── image_time_series_data
        ├── slurm-5085538.out
        ├── test
        ├── test_768.csv
        ├── train
        ├── train_768.csv
        ├── treemonitoring_classes_weights.npy
        └── val_768.csv
        """
        # use the scene flow data folder as input
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        self.month = month


    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'val', 'test'], f"Invalid split name: {split}. Must be one of 'train', 'val', or 'test'."

        with open(os.path.join(self.customized_data_path, f"{split}_768.csv"), 'r') as f:
            lines = f.readlines()

        samples = []
        for line in lines[1:]:
            items = line.strip().split(',')
            for item in items[:-1]:
                if self.month in item:
                    img_file = os.path.join(self.customized_data_path, item)
                    lbl_file = os.path.join(self.customized_data_path, items[-1])
                    if os.path.exists(img_file) and os.path.exists(lbl_file):
                        samples.append((img_file, lbl_file))
                    else:
                        print(f"Warning: Missing img or lbl file for {img_file} and {lbl_file}, Skipping.")

        
        return samples
    
    def _load_original_lbl(self, lbl_file):
        """Load png or tif image file."""
        assert os.path.isfile(lbl_file), f"Label file '{lbl_file}' not found."
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.uint8)[..., 0]  # use only the first channel
        lbl = Image.fromarray(lbl)
        return lbl
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).replace('.png', '')
        zone = img_file.split(os.sep)[-2]
        return f"{zone}_{base_name}"
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(self.color_map)+1)}



    



