# This dataset is processes the ikg ml data for ROI region

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

german_to_english = {
    "wald": "forest",
    "grünland": "grass",
    "siedlung": "settlement",
    "fließgewässer": "water",
    "stillgewässer": "water",
}


class Hismap2Converter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, filtering_keys=None,
                 fg_ratio=0.05, img_size=384, **kwargs):
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
        self.tif_img_dir = os.path.join(customized_data_path, 'tk_image')
        self.tif_lbl_dir = os.path.join(customized_data_path, 'label_tif')
        self.map_files = [x for x in glob(os.path.join(self.tif_img_dir, '*.tif')) if any([fk in x for fk in self.filtering_keys])]
        self.cls_name_to_id = {name: idx for idx, name in self.cls_id_to_name.items()}

    def load_map_and_lbl(self, map_file):

        base_name = os.path.basename(map_file).split('.')[0]
        lbl_files = glob(os.path.join(self.tif_lbl_dir, f'{base_name}_*.tif'))
        label_mask_file = os.path.join(self.tif_lbl_dir, f'{base_name}.tif')
        
        lbl_mask = imread(label_mask_file)
        xs, ys = np.where(lbl_mask == 1)  # foreground pixels
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        img = imread(map_file)
        img = img[xmin:xmax+1, ymin:ymax+1, :]
        lbl_merged = np.zeros_like(lbl_mask[xmin:xmax+1, ymin:ymax+1], dtype=np.uint8)
        for lf in lbl_files:
            cls_name = os.path.basename(lf).split('.')[0].replace(f'{base_name}_', '')
            cls_name = german_to_english[cls_name]
            lbl = imread(lf)
            lbl = lbl[xmin:xmax+1, ymin:ymax+1]
            lbl_merged[lbl == 1] = self.cls_name_to_id[cls_name]
            
        return img, lbl_merged
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).split('.')[0]    
        return base_name    
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(color_map)+1)}
    
    def convert(self, split_list=None):
        self._prepare_convertion()

        # load processed data info if exists
        try:
            data_info = self.load_data_info()
        except:
            data_info = {}

        samples = []
        for map_file in tqdm(self.map_files, desc=f"Converting {self.dataset_name}"):
            img, lbl = self.load_map_and_lbl(map_file)
            img = Image.fromarray(img)
            lbl = self._convert_annotation(lbl)

            crop_w_indices, crop_h_indices = self._get_crop_indices(img.height, img.width)

            for h in crop_h_indices:
                for w in crop_w_indices:
                    base_name = self._parse_base_name(map_file)
                    crop_name = f"{h}_{w}"
                    sample_full_name = f"{self.dataset_name}.{crop_name}"
                    samples.append(sample_full_name)

                    img_crop = img.crop((h, w, h + self.syntra_img_size, w + self.syntra_img_size))
                    img_crop.save(os.path.join(self.img_dir, crop_name + '.png'))
                    if lbl is None:
                        continue
                    lbl_crop = lbl.crop((h, w, h + self.syntra_img_size, w + self.syntra_img_size))
                    lbl_crop.save(os.path.join(self.lbl_dir, crop_name + '.png'))

                    # generate data info
                    data_info[sample_full_name] = self._parse_label_ratios(np.array(lbl_crop))
                        
        self.save_data_info(data_info) # save at finishing every split to avoid data loss
        self.save_samples(samples, 'all')
        self.split_train_test("all_samples.txt", train_ratio=0.5)
    

    




    



