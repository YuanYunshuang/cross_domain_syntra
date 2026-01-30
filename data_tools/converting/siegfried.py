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
  "railway": [139, 69, 19],         
  "vineyard": [0, 100, 0],
}


class SiegfriedConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, class_name, 
                 fg_ratio=0.01, img_size=224, **kwargs):
        """
        Data structure in customized_data_path:
        ├── dataset_railway
            ├── fewshot10
            │   ├── annotation
            │   ├── train
            │   └── val
            ├── full
            │   ├── annotation
            │   ├── test
            │   ├── train
            │   └── val
            ├── partial_1
            │   ├── annotation
            │   ├── train
            │   └── val
            ├── partial_10
            │   ├── annotation
            │   ├── train
            │   └── val
        ├── dataset_vineyard
            ├── fewshot10
            │   ├── annotation
            │   ├── train
            │   └── val
            ├── full
            │   ├── annotation
            │   ├── test
            │   ├── train
            │   └── val

        args:
        class_name: str
            railway or vineyard
        set_name: str   
            The set name to be used, should be one of ['fewshot10', 'full', 'partial_1', 'partial_10'].
        """
        super().__init__(syntra_data_path, 
                        color_map={class_name: color_map[class_name]}, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        self.class_name = class_name
        self.customized_data_path = os.path.join(customized_data_path, f'dataset_{class_name}')
    
    def _load_original_img(self, img_file):
        return Image.fromarray(imread(img_file))
    
    def _load_original_lbl(self, lbl_file):
        lbl = Image.open(lbl_file)
        lbl = (np.array(lbl) > 0).astype(np.uint8)
        return Image.fromarray(lbl)

    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test']
        samples = []
        split_dir = os.path.join(self.customized_data_path, 'full', split)
        for img_file in glob(os.path.join(split_dir, '*.tif')):
            base_name = os.path.basename(img_file)
            lbl_file = os.path.join(self.customized_data_path, 'full', 'annotation', split, base_name)
            samples.append((img_file, lbl_file))

        return samples
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).rsplit('.')[0]   
        return base_name    
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    
    def select_samples(self, set_name):
        """
        Select partial samples from the training dataset.
        """
        data_info = self.load_data_info()
        selected_samples = [x.split('.')[0] for x in os.listdir(
            os.path.join(self.customized_data_path, set_name, 'train')) if x.endswith('.tif')]

        class_counts = {cls_name: 0 for cls_name in self.color_map.keys()}
        for sample in selected_samples:
            class_ratios = data_info[f"{self.class_name}.{sample}"]
            for cls_name, ratio in class_ratios.items():
                class_counts[cls_name] += ratio
        with open(os.path.join(self.syntra_data_path, f'{set_name}_class_distribution.json'), 'w') as f:
            json.dump(class_counts, f, indent=4)
        with open(os.path.join(self.syntra_data_path, f'{set_name}_samples.txt'), 'w') as f:
            for sample in selected_samples:
                f.write(f"{self.class_name}.{sample}" + '\n')  
    

    





    



