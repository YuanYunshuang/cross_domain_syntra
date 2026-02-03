# Convert syntra dataset to version2: all hameln (bayern) data are put into one subset instead of separated folders according to map sheet.

import json
from glob import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from tifffile import imread

from .syntra_data_converter import SyntraDataConverter




class Syntra2Converter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, subset,
                 fg_ratio=0.01, img_size=384, **kwargs):
        """
        Convert the GAUSS dataset to SYNTRA2 format.

        """
        self.syntra1_subsets = [x for x in os.listdir(customized_data_path) if subset in x]
        self.subset = subset
        syntra_data_path = os.path.join(syntra_data_path, subset)
        with open(os.path.join(customized_data_path, self.syntra1_subsets[0], 'color_map.json'), 'r') as f:
            color_map = json.load(f)
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)


    def _parse_split_img_lbl(self, split):
        assert split in ['train', 'test']
        samples = []
        for subset in self.syntra1_subsets:
            split_file = os.path.join(self.customized_data_path, subset, f'{split}_samples.txt')
            with open(split_file, 'r') as f:
                split_samples = f.read().splitlines()
            for sample in split_samples:
                s = sample.split('.')[-1]
                img_file = os.path.join(self.customized_data_path, subset, 'imgs', f'{s}.png')
                lbl_file = os.path.join(self.customized_data_path, subset, 'lbls', f'{s}.png')
                samples.append((img_file, lbl_file))
        return samples

    def select_cls_samples(self, n):
        samples_dict = {}
        for subset in self.syntra1_subsets:
            filename = os.path.join(self.customized_data_path, subset, f'cls_selection_{n}.json')
            with open(filename, 'r') as f:
                cls_samples = json.load(f)
            for cls, samples in cls_samples.items():
                if cls not in samples:
                    samples[cls] = []
                samples_dict[cls].extend(samples)

        output_filename = os.path.join(self.syntra_data_path, subset, f's{n}.json')
        with open(output_filename, 'w') as f:
            json.dump(samples_dict, f, indent=4)
    
    def _load_original_lbl(self, lbl_file):
        """Load tif image file."""
        return Image.open(lbl_file)
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file)
        subset = img_file.split(os.sep)[-3].split('.')[-1]  
        return f'{subset}_{base_name.rsplit(".", 1)[0]}'    
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(self.color_map) + 1)}
    
    def convert(self, split_list=['train', 'test']):
        self._prepare_convertion()

        # load processed data info if exists
        try:
            data_info = self.load_data_info()
        except:
            data_info = {}
        id_to_train_id = self._get_id_map()

        for split in split_list:
            print(f"Processing split: {split} data")
            cur_samples = self._parse_split_img_lbl(split)
            assert len(cur_samples) > 0, f"No samples found for split {split} in {self.customized_data_path}" \
            f". Please check the dataset structure."
            print(f"Found {len(cur_samples)} samples for split {split}. Starting conversion...")

            split_samples = []

            for img_file, lbl_file in tqdm(cur_samples):
                img = self._load_original_img(img_file)
                if lbl_file is None:
                    lbl = None
                else:
                    lbl = self._load_original_lbl(lbl_file)

                base_name = self._parse_base_name(img_file)
                sample_full_name = f"{self.dataset_name}.{base_name}"
                split_samples.append(sample_full_name)

                img.save(os.path.join(self.img_dir, base_name + '.png'))
                if lbl is not None:
                    lbl.save(os.path.join(self.lbl_dir, base_name + '.png'))

                    # generate data info
                    data_info[sample_full_name] = self._parse_label_ratios(np.array(lbl))
                        

            self.save_samples(split_samples, split)
            self.save_data_info(data_info) # save at finishing every split to avoid data loss
    

    





    



