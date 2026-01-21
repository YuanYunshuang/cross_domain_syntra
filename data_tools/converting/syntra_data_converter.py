import os
import json
import random

import numpy as np
from PIL import Image
from tifffile import imread
from matplotlib import pyplot as plt

from indexing.pairing import *



class SyntraDataConverter:
    """
    A class to convert customized dataset to a standardized syntra format.
    """

    def __init__(self, 
                 syntra_data_path: str, 
                 syntra_img_size: int=384,
                 color_map: dict=None, 
                 customized_data_path: str=None, 
                 fg_ratio: float=0.05,
                 force_cast_img_to_uint8: bool=False,
        ):
        """
        Initializes the SyntraDataConverter with paths to the customized dataset and Syntra dataset.

        Args:
            syntra_data_path (str): Path to the Syntra dataset.
            color_map (dict, optional): A mapping of class names to colors. Defaults to None.
            customized_data_path (str, optional): Path to the customized dataset. Defaults to None.
            fg_ratio (float, optional): Minimum foreground ratio for a class to be considered present in a sample. Defaults to 0.05.
            syntra_img_size (int, optional): The size of the images in the Syntra dataset. Defaults to 384.
            force_cast_img_to_uint8 (bool, optional): Whether to force cast images to uint8. Defaults to False.
        """
        self.customized_data_path = customized_data_path
        self.syntra_data_path = syntra_data_path
        self.dataset_name = os.path.basename(syntra_data_path.rstrip('/'))
        self.fg_ratio = fg_ratio
        self.syntra_img_size = syntra_img_size
        self.force_cast_img_to_uint8 = force_cast_img_to_uint8

        self.prepare_img_lbl_dirs()

        # if the data is already converted the color map is available
        color_map_path = os.path.join(syntra_data_path, "color_map.json")
        if color_map is None:
            assert os.path.isfile(color_map_path), f"Color map file not found at {color_map_path}, please provide cls_names."
            with open(color_map_path, 'r') as f:
                self.color_map = json.load(f)
                self.color_map = {k: v for k, v in self.color_map.items() if k not in ['background', 'nonlabeled']}
        else:
            # initialize color map for the customized dataset
            assert customized_data_path is not None, "Customized dataset path and cls_names should both be given to initialize the converter for the first time."
            for cls_name, color in color_map.items():
                if color is None:
                    color_map[cls_name] = [random.randint(0, 255) for _ in range(3)]
            # save color map
            self.color_map = color_map
            with open(color_map_path, 'w') as f:
                json.dump(self.color_map, f, indent=4)

        self._construct_color_palette()
        self.generate_legend_img()
    
    @property
    def cls_id_to_name(self):
        return {i+1: name for i, name in enumerate(self.color_map.keys())}
    
    def prepare_img_lbl_dirs(self):
        """
        Prepares the image and label directories within the Syntra dataset path.
        """
        self.img_dir = os.path.join(self.syntra_data_path, "imgs")
        self.lbl_dir = os.path.join(self.syntra_data_path, "lbls")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
    
    def _construct_color_palette(self):
        """
        Constructs a color palette list from the color map.
        Returns:
            list: A flat list representing the color palette.
        """
        palette = [0, 0, 0]  # background color
        for cls_name in self.color_map.keys():
            palette.extend(self.color_map[cls_name])
        # pad the palette to have 256 colors
        palette.extend([0, 0, 0] * (256 - len(self.color_map) - 1))
        # set the last color to white as nonlabled
        palette[-3:] = [255, 255, 255]
        self.palette = palette

    def generate_legend_img(self):
        legend_img_path = os.path.join(self.syntra_data_path, "legend.png")

        # draw legend image using color_map
        n_colors = len(self.color_map)
        _, ax = plt.subplots(figsize=(3, n_colors * 0.3))
        for i, (cls_name, color) in enumerate(self.color_map.items()):
            ax.barh(i, 1, color=np.array(color)/255.0)
            ax.text(0.5, i, f"{i+1}-{cls_name}", va='center', ha='center', color='black' if np.mean(color) > 128 else 'white')
        ax.set_yticks([])
        ax.set_xticks([])
        plt.savefig(legend_img_path, bbox_inches='tight')
    
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

    def _parse_split_img_lbl(self, split):
        raise NotImplementedError("This method should be implemented in subclasses." \
        "It should return a list of (img_file, lbl_file) tuples for the given split.")
    
    def _get_crop_indices(self, width, height, overide_crop_size=None):
        """
        The default cropping strategy is to use non-overlapping crops.
        The last crop in each row/column may overlap with the previous one to ensure full coverage.
        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            overide_crop_size (int, optional): Crop size to override the default syntra_img_size. Defaults to None.
        Returns:
            tuple: Two lists containing the x and y indices for cropping.
        """
        crop_size = self.syntra_img_size if overide_crop_size is None else overide_crop_size
        w_indices = list(range(0, width, crop_size))
        h_indices = list(range(0, height, crop_size))
        if w_indices[-1] + crop_size > width:
            w_indices[-1] = width - crop_size
        if h_indices[-1] + crop_size > height:
            h_indices[-1] = height - crop_size
        return w_indices, h_indices

    def _load_original_img(self, img_file):
        """Load png or tif image file."""
        assert os.path.isfile(img_file), f"Image file '{img_file}' not found."
        if img_file.endswith('.png'):
            img = Image.open(img_file).convert("RGB")
        elif img_file.endswith('.tif') or img_file.endswith('.tiff'):
            img = imread(img_file)
            if img.dtype != np.uint8 and self.force_cast_img_to_uint8:
                if img.dtype == np.uint16:
                    print(f"Warning: casting image '{img_file}' from {img.dtype} to uint8.")
                    img = img.astype(np.uint8)
                elif img.dtype == np.int16:
                    assert img.min() >= 0, f"Image '{img_file}' has negative values, cannot cast to uint8."
                    print(f"Warning: casting image '{img_file}' from {img.dtype} to uint8.")
                    img = img.astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported image dtype '{img.dtype}' for file '{img_file}'.")
            else:
                assert img.dtype == np.uint8, f"Image dtype must be uint8, got {img.dtype}."

            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            img = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image format for file '{img_file}'.")
        return img
    
    def _load_original_lbl(self, lbl_file):
        """Load png or tif image file."""
        assert os.path.isfile(lbl_file), f"Label file '{lbl_file}' not found."
        if lbl_file.endswith('.png'):
            lbl = Image.open(lbl_file)
        elif lbl_file.endswith('.tif') or lbl_file.endswith('.tiff'):
            lbl = imread(lbl_file)
            assert lbl.dtype == np.uint8, f"Label dtype must be uint8, got {lbl.dtype}."
            assert lbl.ndim == 2, f"Label image must be 2D, got shape {lbl.shape}."
            lbl = Image.fromarray(lbl)
        else:
            raise ValueError(f"Unsupported image format for file '{lbl_file}'.")
        return lbl
    
    def _parse_base_name(self, img_file):
        raise NotImplementedError("This method should be implemented in subclasses." \
        "It should return the base name (new formatting for syntra) of the image file without extensions or suffixes.")
    
    def _get_id_map(self):
        raise NotImplementedError("This method should be implemented in subclasses." \
        "It should return a mapping from original class ids to new class ids.")
    
    def _parse_label_ratios(self, lbl_array):
        total_pixels = lbl_array.size
        class_ratios = {}
        for cls_id, cls_name in self.cls_id_to_name.items():
            cls_pixels = np.sum(lbl_array == cls_id)
            class_ratios[cls_name] = cls_pixels / total_pixels
        return class_ratios
    
    def _parse_data_info(self, save=False):
        lbl_files = glob(os.path.join(self.lbl_dir, "*.png"))
        data_info = {}
        for lbl_file in lbl_files:
            full_name = f"{self.dataset_name}.{os.path.basename(lbl_file).rsplit('.', 1)[0]}"
            data_info[full_name] = self._parse_label_ratios(np.array(Image.open(lbl_file)))
        if save:
            self.save_data_info(data_info)
        return data_info
    
    def convert(self, split_list=['train', 'test', 'val']):
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
                    lbl = self._convert_annotation(np.array(lbl), id_to_train_id)

                crop_w_indices, crop_h_indices = self._get_crop_indices(img.height, img.width)

                for h in crop_h_indices:
                    for w in crop_w_indices:
                        base_name = self._parse_base_name(img_file)
                        crop_name = f"{base_name}_{h}_{w}"
                        sample_full_name = f"{self.dataset_name}.{crop_name}"
                        split_samples.append(sample_full_name)

                        img_crop = img.crop((h, w, h + self.syntra_img_size, w + self.syntra_img_size))
                        img_crop.save(os.path.join(self.img_dir, crop_name + '.png'))
                        if lbl is None:
                            continue
                        lbl_crop = lbl.crop((h, w, h + self.syntra_img_size, w + self.syntra_img_size))
                        lbl_crop.save(os.path.join(self.lbl_dir, crop_name + '.png'))

                        # generate data info
                        data_info[sample_full_name] = self._parse_label_ratios(np.array(lbl_crop))
                        

            self.save_samples(split_samples, split)
            self.save_data_info(data_info) # save at finishing every split to avoid data loss

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
                if cls_name in ['background', 'nonlabeled']:
                    continue
                class_counts[cls_name] += ratio
        with open(os.path.join(self.syntra_data_path, f'{split}_class_distribution.json'), 'w') as f:
            json.dump(class_counts, f, indent=4)

    def split_train_test(self, sample_file, train_ratio=0.5):
        """
        Splits the dataset into training and testing sets.

        Args:
            sample_file (str): Path to the file containing all sample names.
            train_ratio (float): Ratio of samples to be used for training.
        """
        with open(os.path.join(self.syntra_data_path, sample_file), 'r') as f:
            all_samples = f.readlines()
        all_samples = [line.strip() for line in all_samples]
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        self.save_samples(train_samples, split="train")
        self.save_samples(test_samples, split="test")

        self.summarize_class_distribution(split="train")
        self.summarize_class_distribution(split="test")
    
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
    
    def select_cls_samples(self, n, sample_file=None):
        """
        Select n samples for each class from the training dataset.
        """
        sample_file = sample_file if sample_file is not None else "train_samples.txt"
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
            if cls_name in ['background', 'nonlabeled']:
                continue
            if len(samples) == 0:
                print(f"Warning: No samples found for class '{cls_name}'. Skip.")
                continue
            selected = random.sample(samples, min(n, len(samples)))
            selected_samples[cls_name] = selected
        
        # save the selection
        with open(os.path.join(self.syntra_data_path, f"cls_selection_{n}.json"), 'w') as f:
            json.dump(selected_samples, f, indent=4)

    def pairing_intra_cls(self, tgt_info_file, src_info_file=None, k=8, skip_if_lt_k=False):
      
        if src_info_file is None:
            src_info_file = tgt_info_file

        img_encoder = DinoV2Encoder()
        class_names = self.color_map.keys()
        tgt_samples = load_samples(os.path.join(self.syntra_data_path, tgt_info_file), keep_cls_structure=True)
        src_samples = load_samples(os.path.join(self.syntra_data_path, src_info_file), keep_cls_structure=True)
        ext = "." + os.listdir(self.img_dir)[0].rsplit('.', 1)[-1]
        if isinstance(src_samples, list):
            src_samples = parse_class_samples(self.load_data_info(), src_samples, list(self.color_map.keys()), fg_ratio=self.fg_ratio)
        if not 'test' in tgt_info_file and isinstance(tgt_samples, list):
            # for test samples, visible classes are not known, we pair for all classes
            tgt_samples = parse_class_samples(self.load_data_info(), tgt_samples, list(self.color_map.keys()), fg_ratio=self.fg_ratio)
        
        pairing_info = {}
        for cls_name in class_names:
            if cls_name not in src_samples or len(src_samples[cls_name]) == 0:
                print(f"Warning: No source samples found for class {cls_name}. Skip.")
                continue
            print('*' * 30 + f"Processing class: {cls_name}")
            pairing_info[cls_name] = {}
            cur_src_samples = src_samples[cls_name]
            cur_tgt_samples = tgt_samples[cls_name] if isinstance(tgt_samples, dict) else tgt_samples
            if len(cur_src_samples) < 8:
                if skip_if_lt_k:
                    print(f"Warning: class {cls_name} has less than 8 source samples. Skip pairing.")
                    continue
                else:
                    print(f"Warning: class {cls_name} has less than 8 source samples. Select all.")

            # build index db for current source samples
            map_dataloader = BaseDataloaders(os.path.dirname(self.syntra_data_path), cur_src_samples, batch_size=4, ext=ext)
            index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)
    
            for sample in tqdm(cur_tgt_samples):
                sample_name = sample.rsplit('.', 1)[1] 
                sample_img_path = os.path.join(self.img_dir, f"{sample_name}{ext}")
                distances, indices =  index_db.external_search(
                    self.dataset_name, sample_img_path, k=min(k, len(cur_src_samples))) 
                # remove self from the pairing list
                mask = distances>1e-3
                distances = distances[mask]
                indices = indices[mask]
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
    
    def pairing_sample_intra_cls(self, tgt_info_file, src_info_file=None, k=8, skip_if_lt_k=False):
        """
        pairing format: 
            target_sample: {
                "cls_name": {
                    "src_samples": [list of source samples],
                }
        """
      
        if src_info_file is None:
            src_info_file = tgt_info_file

        img_encoder = DinoV2Encoder()
        class_names = self.color_map.keys()
        tgt_samples = load_samples(os.path.join(self.syntra_data_path, tgt_info_file))
        src_samples = load_samples(os.path.join(self.syntra_data_path, src_info_file))
        ext = "." + os.listdir(self.img_dir)[0].rsplit('.', 1)[-1]
        if isinstance(src_samples, list):
            src_samples = parse_class_samples(self.load_data_info(), src_samples, list(self.color_map.keys()), fg_ratio=self.fg_ratio)
        
        pairing_info = {x: {} for x in tgt_samples}
        for cls_name in class_names:
            if cls_name not in src_samples or len(src_samples[cls_name]) == 0:
                print(f"Warning: No source samples found for class {cls_name}. Skip.")
                continue
            print('*' * 30 + f"Processing class: {cls_name}")
            cur_src_samples = src_samples[cls_name]
            cur_tgt_samples = tgt_samples
            if len(cur_src_samples) < 8:
                if skip_if_lt_k:
                    print(f"Warning: class {cls_name} has less than 8 source samples. Skip pairing.")
                    continue
                else:
                    print(f"Warning: class {cls_name} has less than 8 source samples. Select all.")

            # build index db for current source samples
            map_dataloader = BaseDataloaders(os.path.dirname(self.syntra_data_path), cur_src_samples, batch_size=4, ext=ext)
            index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)
    
            for sample in tqdm(cur_tgt_samples):
                sample_name = sample.rsplit('.', 1)[1] 
                sample_img_path = os.path.join(self.img_dir, f"{sample_name}{ext}")
                distances, indices =  index_db.external_search(
                    self.dataset_name, sample_img_path, k=min(k, len(cur_src_samples))) 
                # remove self from the pairing list
                mask = distances>1e-3
                distances = distances[mask]
                indices = indices[mask]
                # update pairing info
                pairing_info[sample][cls_name] = {
                    'src_samples': [cur_src_samples[i] for i in indices],
                    # 'distances': distances.tolist()
                }
            del index_db
            del map_dataloader

        # save pairing info
        src_mode = os.path.splitext(src_info_file)[0]
        tgt_mode = os.path.splitext(tgt_info_file)[0]
        save_path = os.path.join(self.syntra_data_path, f'sample_pair_{src_mode}_{tgt_mode}_intracls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)

    def pairing_negative_samples(self, tgt_info_file, src_info_file, k=8):
        """
        For negative samples which do not contain any foreground class, we pair them with random samples from the source dataset.
        Args:
            tgt_info_file (str): Target dataset sample info file.
            src_info_file (str): Source dataset sample info file.
            k (int, optional): Number of samples to pair. Defaults to 8.
        """
        
        src_mode = os.path.splitext(src_info_file)[0]
        tgt_mode = os.path.splitext(tgt_info_file)[0]
        pairing_info_path = os.path.join(self.syntra_data_path, f'pairing_{src_mode}_{tgt_mode}_intracls.json')

        with open(pairing_info_path, 'r') as f:
            pairing_info = json.load(f)
        src_samples = load_samples(os.path.join(self.syntra_data_path, src_info_file), remove_repeat=True)
        tgt_samples = load_samples(os.path.join(self.syntra_data_path, tgt_info_file), remove_repeat=True)
        for sample in tgt_samples:
            if any(sample in pairing_info.get(cls_name, {}) for cls_name in self.color_map.keys()):
                continue  # already has pairing info

            # randomly select k samples from src_samples
            selected_src_samples = random.sample(src_samples, min(k, len(src_samples)))
            pairing_info.setdefault('negative', {})[sample] = {
                'src_samples': selected_src_samples,
                'distances': [None] * len(selected_src_samples)
            }

        # save updated pairing info
        with open(pairing_info_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)
