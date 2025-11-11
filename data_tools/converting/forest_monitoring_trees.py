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
}



class ForestMonitoringTreesConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path=None, fg_ratio=0.05):
        if customized_data_path is None:
            super().__init__(syntra_data_path, fg_ratio=fg_ratio)
        else:
            cls_names = list(color_map.keys())
            cls_colors = [color_map[cls_name] for cls_name in cls_names]
            super().__init__(syntra_data_path, 
                            cls_names=cls_names, 
                            cls_colors=cls_colors,
                            customized_data_path=customized_data_path, 
                            fg_ratio=fg_ratio)


    def convert(self, ):
        self._prepare_convertion()
        input_dir = self.customized_data_path
        output_dir = self.syntra_data_path

        data_info = {}
        cls_id_to_name = {i+1: name for i, name in enumerate(color_map.keys())}

        for split in ['train', 'test']:
            cur_samples = []
            input_split_dir = os.path.join(input_dir, split)
            img_files = glob(os.path.join(input_split_dir, "images", "*/*.png"))

            for img_file in img_files:
                print(img_file)
                base_name = os.path.basename(img_file)
                zone = img_file.split('/')[-2]
                label_file = img_file.replace("images", "labels_merged")

                if not os.path.exists(label_file):
                    continue

                img = Image.open(img_file).convert("RGB")
                lbl = np.array(Image.open(label_file))
                lbl = self._convert_annotation(lbl)

                s = self.syntra_img_size
                for x in range(img.width // s):
                    for y in range(img.height // s):
                        img_crop = img.crop((x * s, y * s, (x + 1) * s, (y + 1) * s))
                        lbl_crop = lbl.crop((x * s, y * s, (x + 1) * s, (y + 1) * s))
                        crop_name = f"{base_name.replace('.png', '')}_{zone}_{x}_{y}"
                        img_crop.save(os.path.join(output_dir, 'imgs', crop_name + '.png'))
                        lbl_crop.save(os.path.join(output_dir, 'lbls', crop_name + '.png'))
                        cur_samples.append(f"forest_monitoring.trees.{crop_name}")

                        # generate data info
                        lbl_array = np.array(lbl_crop)
                        total_pixels = lbl_array.size
                        class_ratios = {}
                        for cls_id, cls_name in cls_id_to_name.items():
                            class_pixels = np.sum(lbl_array == cls_id)
                            class_ratios[cls_name] = class_pixels / total_pixels
                        data_info[f"forest_monitoring.trees.{crop_name}"] = class_ratios
            self.save_samples(cur_samples, split)
        self.save_data_info(data_info)



    



