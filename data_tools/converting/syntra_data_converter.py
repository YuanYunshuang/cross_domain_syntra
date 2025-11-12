import os
import json
import random

import numpy as np
from PIL import Image

from indexing.pairing import *



class SyntraDataConverter:
    """
    A class to convert customized dataset to a standardized syntra format.
    """

    def __init__(self, 
                 syntra_data_path, 
                 syntra_img_size=384,
                 cls_names=None, 
                 cls_colors=None, 
                 customized_data_path=None, 
                 fg_ratio=0.05):
        """
        Initializes the SyntraDataConverter with paths to the customized dataset and Syntra dataset.

        Args:
            syntra_data_path (str): Path to the Syntra dataset.
            color_map (dict, optional): A mapping of class names to colors. Defaults to None.
            customized_data_path (str, optional): Path to the customized dataset. Defaults to None.
        """
        self.customized_data_path = customized_data_path
        self.syntra_data_path = syntra_data_path
        self.dataset_name = os.path.basename(syntra_data_path.rstrip('/'))
        self.fg_ratio = fg_ratio
        self.syntra_img_size = syntra_img_size
        # if the data is already converted the color map is available
        color_map_path = os.path.join(syntra_data_path, "color_map.json")
        if cls_names is None:
            assert os.path.isfile(color_map_path), f"Color map file not found at {color_map_path}, please provide cls_names."
            with open(color_map_path, 'r') as f:
                self.color_map = json.load(f)
        else:
            # initialize color map for the customized dataset
            assert customized_data_path is not None, "Customized dataset path and cls_names should both be given to initialize the converter for the first time."
            if cls_colors is None:
                # generate random colors for each class
                self.color_map = {cls_name: [random.randint(0, 255) for _ in range(3)] for cls_name in cls_names}
                # save color map
                with open(color_map_path, 'w') as f:
                    json.dump(self.color_map, f, indent=4)

        self._construct_color_palette()
        self.img_dir = os.path.join(self.syntra_data_path, "imgs")
        self.lbl_dir = os.path.join(self.syntra_data_path, "lbls")
    
    def _construct_color_palette(self):
        """
        Constructs a color palette list from the color map.
        Returns:
            list: A flat list representing the color palette.
        """
        palette = [0, 0, 0]  # background color
        for cls_name in sorted(self.color_map.keys()):
            palette.extend(self.color_map[cls_name])
        # pad the palette to have 256 colors
        palette.extend([0, 0, 0] * (256 - len(self.color_map) - 1))
        # set the last color to white as nonlabled
        palette[-3:] = [255, 255, 255]
        self.palette = palette
    
    def _convert_annotation(self, lbl_array, id_map=None):
        """
        Converts the label image according to the provided id_map if given, otherwise it will be mapped with 1->1, 2->2, ...

        Args:
            lbl (np.ndarray): The label image to be converted.
            id_map (dict): A mapping from original class ids to new class ids.

        Returns:
            PIL.Image: The converted label image.
        """
        if id_map is None:
            id_map = {i: i for i in range(1, len(self.color_map) + 1)}
        converted_lbl = np.zeros_like(lbl_array, dtype=np.uint8)
        for orig_id, new_id in id_map.items():
            converted_lbl[lbl_array == orig_id] = new_id
        
        converted_lbl = Image.fromarray(converted_lbl, mode='P')
        converted_lbl.putpalette(self.palette)
        return converted_lbl
    
    def _prepare_convertion(self):
        """
        Initializes the necessary directories for the Syntra dataset.
        """
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        assert self.customized_data_path is not None, "Customized dataset path must be provided."

    def save_samples(self, samples, split):
        """
        Saves the list of samples to a text file for the given split.

        Args:
            samples (list): List of sample names.
            split (str): The dataset split ('train' or 'test').
        """
        samples_file = os.path.join(self.syntra_data_path, f"{split}_samples.txt")
        with open(samples_file, 'w') as f:
            for sample in samples:
                f.write(sample + '\n')
                
    def load_samples(self, split):
        """
        Loads the list of samples from a text file for the given split.

        Args:
            split (str): The dataset split ('train' or 'test').

        Returns:
            list: List of sample names.
        """
        samples_file = os.path.join(self.syntra_data_path, f"{split}_samples.txt")
        assert os.path.isfile(samples_file), f"Samples file '{samples_file}' not found."
        with open(samples_file, 'r') as f:
            samples = f.readlines()
        samples = [line.strip() for line in samples]
        return samples
    
    def save_data_info(self, data_info):
        """
        Saves the data information to a JSON file.

        Args:
            data_info (dict): A dictionary containing data information.
        """
        data_info_file = os.path.join(self.syntra_data_path, f"{self.dataset_name}.json")
        with open(data_info_file, 'w') as f:
            json.dump(data_info, f, indent=4)
    
    def load_data_info(self):
        """
        Loads the data information from a JSON file.

        Returns:
            dict: A dictionary containing data information.
        """
        data_info_file = os.path.join(self.syntra_data_path, f"{self.dataset_name}.json")
        assert os.path.isfile(data_info_file), f"Data info file '{data_info_file}' not found."
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)
        return data_info
    
    def summarize_class_distribution(self, split="train"):
        # original distribution
        class_counts = {cls_name: 0 for cls_name in self.color_map.keys()}
        all_samples = self.load_samples(split)
        data_info = self.load_data_info()
        for sample in all_samples:
            if sample not in data_info:
                print(f"Warning: {sample} not in data info.")
            class_ratios = data_info[sample]
            for cls_name, ratio in class_ratios.items():
                class_counts[cls_name] += ratio
        with open(os.path.join(self.syntra_data_path, f'{split}_class_distribution.json'), 'w') as f:
            json.dump(class_counts, f, indent=4)


    def convert(self):
        """
        Converts Syntra dataset annotations to a standardized format.
        """
        
        raise NotImplementedError("This method should be implemented to convert customized dataset to Syntra format.")
    
    def select_samples(self, n, enhance_sample_balance=False):
        """
        Select n samples from the training dataset.
        """
        train_samples_file = os.path.join(self.syntra_data_path, "train_samples.txt")
        assert os.path.isfile(train_samples_file), "Training samples, 'train_samples.txt' file not found."

        with open(train_samples_file, 'r') as f:
            train_samples = f.readlines()
        train_samples = [line.strip() for line in train_samples]

        if not enhance_sample_balance:
            selected_samples = random.sample(train_samples, min(n, len(train_samples)))

            # save the selection
            with open(os.path.join(self.syntra_data_path, f"train_selection_{n}.txt"), 'w') as f:
                for sample in selected_samples:
                    f.write(sample + '\n')
        else:
            # class-balanced selection
            selected_samples = []

            data_info = self.load_data_info()
            cls_to_samples = {k: [] for k in self.color_map.keys()}
            for sample in train_samples:
                class_ratios = data_info[sample]
                for cls_name, ratio in class_ratios.items():
                    cls_to_samples[cls_name].append((sample, ratio))
            for cls_name, samples_ratios in cls_to_samples.items():
                if len(samples_ratios) == 0:
                    continue
                selected_samples.extend(np.random.choice(
                    [sr[0] for sr in samples_ratios],
                    size=max(1, n),
                    replace=False
                ).tolist())
            
            # remove repeat samples
            selected_samples = list(set(selected_samples))

        class_counts = {cls_name: 0 for cls_name in self.color_map.keys()}
        for sample in selected_samples:
            class_ratios = data_info[sample]
            for cls_name, ratio in class_ratios.items():
                class_counts[cls_name] += ratio
        with open(os.path.join(self.syntra_data_path, f'train_selection_{n}_class_distribution.json'), 'w') as f:
            json.dump(class_counts, f, indent=4)
        with open(os.path.join(self.syntra_data_path, f'train_selection_{n}.txt'), 'w') as f:
            for sample in selected_samples:
                f.write(sample + '\n')
    
    def select_cls_samples(self, n):
        """
        Select n samples for each class from the training dataset.
        """
        train_samples_file = os.path.join(self.syntra_data_path, "train_samples.txt")
        assert os.path.isfile(train_samples_file), "Training samples, 'train_samples.txt' file not found."

        with open(train_samples_file, 'r') as f:
            train_samples = f.readlines()
        train_samples = [line.strip() for line in train_samples]

        data_info_file = os.path.join(self.syntra_data_path, f"{self.dataset_name}.json")
        assert os.path.isfile(data_info_file), f"Data info file '{data_info_file}' not found."
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)

        cls_to_samples = {cls_name: [] for cls_name in self.color_map.keys()}
        for sample in train_samples:
            class_ratios = data_info.get(sample, {})
            for cls_name, cls_ratio in class_ratios.items():
                if cls_name in cls_to_samples and cls_ratio > self.fg_ratio:
                    cls_to_samples[cls_name].append(sample)
        
        # select n samples for each class
        selected_samples = {}
        for cls_name, samples in cls_to_samples.items():
            if len(samples) == 0:
                print(f"Warning: No samples found for class '{cls_name}'. Skip.")
                continue
            selected = random.sample(samples, min(n, len(samples)))
            selected_samples[cls_name] = selected
        
        # save the selection
        with open(os.path.join(self.syntra_data_path, f"cls_selection_{n}.json"), 'w') as f:
            json.dump(selected_samples, f, indent=4)

    def pairing_intra_cls(self, tgt_info_file, src_info_file=None, k=8):
        if src_info_file is None:
            src_info_file = tgt_info_file

        img_encoder = DinoV2Encoder()
        class_names = self.color_map.keys()
        tgt_samples = load_samples(os.path.join(self.syntra_data_path, tgt_info_file), keep_cls_structure=True)
        src_samples = load_samples(os.path.join(self.syntra_data_path, src_info_file), keep_cls_structure=True)
        ext = "." + os.listdir(self.img_dir)[0].rsplit('.', 1)[-1]
        if isinstance(src_samples, list):
            src_samples = parse_class_samples(self.load_data_info(), src_samples, list(self.color_map.keys()))
        if not 'test' in tgt_info_file and isinstance(tgt_samples, list):
            # for test samples, visible classes are not known, we pair for all classes
            tgt_samples = parse_class_samples(self.load_data_info(), tgt_samples, list(self.color_map.keys()))
        
        pairing_info = {}
        for cls_name in class_names:
            if cls_name not in src_samples or len(src_samples[cls_name]) == 0:
                print(f"Warning: No source samples found for class {cls_name}. Skip.")
                continue
            print(f"Processing class: {cls_name}")
            pairing_info[cls_name] = {}
            cur_src_samples = src_samples[cls_name]
            cur_tgt_samples = tgt_samples[cls_name] if isinstance(tgt_samples, dict) else tgt_samples
            # build index db for current source samples
            map_dataloader = BaseDataloaders(os.path.dirname(self.syntra_data_path), cur_src_samples, batch_size=4, ext=ext)
            index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)
    
            for sample in tqdm(cur_tgt_samples):
                sample_name = sample.rsplit('.', 1)[1] 
                sample_img_path = os.path.join(self.img_dir, f"{sample_name}{ext}")
                distances, indices =  index_db.external_search(
                    self.dataset_name, sample_img_path, k=k) 
                # remove self from the pairing list
                mask = distances>1e-3
                distances = distances[mask]
                indices = indices[mask]
                if len(indices) < k - 1:
                    print(f"Warning: sample {sample} has less than {k-1} neighbors. Final selected neighbors: {len(indices)}")
                # update pairing info
                pairing_info[cls_name][sample] = {
                    'src_samples': [cur_src_samples[i] for i in indices],
                    'distances': distances.tolist()
                }
            del index_db
            del map_dataloader

        # save pairing info
        src_mode = os.path.splitext(src_info_file)[0]
        tgt_mode = os.path.splitext(tgt_info_file)[0]
        save_path = os.path.join(self.syntra_data_path, f'pairing_{src_mode}_{tgt_mode}_intracls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)
