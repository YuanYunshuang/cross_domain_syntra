# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import os
import sys
import logging
import random
import argparse
import shutil
from os.path import join
from tqdm import tqdm
from glob import glob
import json

    
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


from syntra.modeling import dataset
from syntra.modeling.dataset import get_dataset
from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling.losses import get_loss_fn
from syntra.utils.vis import vis_train
from syntra.modeling.validator import Validator
from syntra.modeling import build_segmentation_model
from syntra.utils.train_utils import adjust_learning_rate_cosine, adjust_learning_rate_exp_decay
from syntra.utils.sampler import get_sampler
from syntra.utils.debug_utils import *
from data_tools.indexing.pairing import *
from data_tools.indexing.faiss_utils import FAISSIndexer

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def set_seeds(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path



def parse_args():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="sub-commands: inference or test")
    
    # sub-commands for test
    parser_test = subparsers.add_parser("test", help="inference")
    parser_test.add_argument("-d", "--data_root", type=str, help="path to dataset root directory")
    parser_test.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_test.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_test.add_argument("-c", "--config", type=str, help="config file name")
    parser_test.add_argument("-n", "--num-src", type=int, help="number of source images")

    # sub-commands for inference
    parser_inference = subparsers.add_parser("inference", help="inference")
    parser_inference.add_argument("-s", "--source-data-dir", type=str, help="path to source dataset root directory")
    parser_inference.add_argument("-t", "--target-data-dir", type=str, help="path to target dataset root directory")
    parser_inference.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_inference.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_inference.add_argument("-c", "--config", type=str, help="config file name")
    parser_inference.add_argument("-n", "--num-src", type=int, help="number of source images")

    args = parser.parse_args()

    return args


class BaseSourceDataset(Dataset):
    def __init__(self, data_root, filenames, ext='.png', normalize=True):
        self.ext = ext
        self.normalize = normalize
        self.filenames = [f.split('.')[-1] + ext for f in filenames]  
        self.data_root = os.path.join(data_root, 'imgs') 

        # Define preprocessing for the image
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to 224x224
                transforms.ToTensor(),          # Convert to tensor
                transforms.Normalize(           # Normalize using ImageNet mean and std
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
            ]
        )

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        full_path = os.path.join(self.data_root, filename)

        image = Image.open(full_path).convert('RGB')
        image = self.preprocess(image)  
        return full_path, image
    
    def collate_fn(self, batch):
        paths, images = zip(*batch)
        images = torch.stack(images, dim=0)
        return paths, images


def build_source_dataloader(data_root, sample_list_file, batch_size=32, shuffle=False, num_workers=4, ext='.png', normalize=True):
    with open(sample_list_file, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    dataset = BaseSourceDataset(data_root, filenames, ext, normalize=normalize)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return dataloader
    

class SourceIndexDatabase:
    def __init__(self, 
                 db_path, 
                 source_sample_list_file,
                 index_dim=384, 
                 cache_original_vec=True):
        """
        Docstring for __init__
        
        :param self: Instance of the IndexDatabase class
        :param db_path: Database path which contains structured data subsets, each subset contains imgs and lbls dir, and the source_sample_list_file
        :param source_sample_list_file: Source sample list file for indexing, they should be relative paths under each subset imgs dir.
        :param index_dim: Dimension of the index vectors
        :param cache_original_vec: Cache the original vectors while indexing or not
        """
        self.db_path = db_path
        self.source_sample_list_file = source_sample_list_file
        self.index_dim = index_dim
        self.cache_original_vec = cache_original_vec
        self.img_index = {}
        self.encoded_vectors = {}
        self.subset_dirs = {}
        self.map_idx_to_path = {}
        self.map_idx_to_name = {}
        self.map_name_to_idx = {}

        self.src_preprocess = BaseSourceDataset("", []).preprocess

        self.build_img_encoder()
        self.build_index_database()

    @property
    def datasets(self):
        return list(self.img_index.keys())
    
    def build_img_encoder(self):
        self.img_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', source='github')
        self.img_encoder.cuda().eval()  # Set the model to evaluation mode
    
    # def build_dataloaders(self):
    #     # parse subset dirs
    #     self.dataloaders = {}
    #     subset_dirs = [os.path.dirname(d) for d in glob(os.path.join(self.db_path, f'**/{self.source_sample_list_file}'))]
    #     for subset in subset_dirs:
    #         subset_name = os.path.basename(subset)
    #         dataloader = build_source_dataloader(data_root=subset, 
    #                                              sample_list_file=os.path.join(subset, self.source_sample_list_file),
    #                                              batch_size=32,
    #                                              shuffle=False,
    #                                              num_workers=4,
    #                                              ext='.png',
    #                                              normalize=True)
    #         self.dataloaders[subset_name] = dataloader
    
    def generate_embedding(self, input_tensor):
        # Generate embeddings
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.img_encoder.forward_features(input_tensor)

        return embeddings
    
    def indexing_subset(self, subset):
        """
        Index a single subset dataset.
        
        :param subset: Path of the subset to be indexed.
        """
        subset_name = os.path.basename(subset)
        self.subset_dirs[subset_name] = subset
        dataloader = build_source_dataloader(data_root=subset, 
                                             sample_list_file=os.path.join(subset, self.source_sample_list_file))

        faiss_index = FAISSIndexer(dimension=384)  # DINOv2 Vit-S/14 embedding dimension
        cur_paths = []
        cur_img_vectors = []
        pbar = tqdm(dataloader)
        
        for paths, imgs in pbar:
            imgs = imgs.cuda()
            emb_dict = self.generate_embedding(imgs)
            token = emb_dict['x_norm_clstoken']  # (B, C)
            cur_paths.extend(paths)
            faiss_index.add(token)
            if self.cache_original_vec:
                cur_img_vectors.append(token)
            pbar.set_postfix({'Processing dataset': subset_name, 'Images indexed': len(cur_paths)})
        
        self.img_index[subset_name] = faiss_index
        if self.cache_original_vec:
            self.encoded_vectors[subset_name] = torch.cat(cur_img_vectors, dim=0)
        self.map_idx_to_path[subset_name] = cur_paths
        self.map_idx_to_name[subset_name] = {
            i: os.path.basename(p).split('.')[0] for i, p in enumerate(cur_paths)}
        self.map_name_to_idx[subset_name] = {
            k: v for v, k in self.map_idx_to_name[subset_name].items()}

    def build_index_database(self):
        subset_dirs = [os.path.dirname(d) for d in glob(os.path.join(self.db_path, f'**/{self.source_sample_list_file}'))]
        for subset in subset_dirs:
            if self.load_index(subset):
                subset_name = os.path.basename(subset)
                self.subset_dirs[subset_name] = subset
                print(f"Loaded index for data subset: {subset}")
            else:
                self.indexing_subset(subset)
                self.save_index(subset)
                print(f"Indexed and saved data subset: {subset}")
    
    def index_filename(self, subset_path):
        return os.path.join(subset_path, f"{self.source_sample_list_file.split('.')[0]}.faiss")
    
    def index_map_filename(self, subset_path):
        return os.path.join(subset_path, f"{self.source_sample_list_file.split('.')[0]}_map.json")
    
    def save_index(self, subset_path):
        subset_name = os.path.basename(subset_path)

        faiss_index = self.img_index[subset_name]
        cur_paths = self.map_idx_to_path[subset_name]
        if faiss_index is None or len(cur_paths) == 0:
            print(f"No index or paths found for dataset {subset_name}, skipping save.")
            return

        # Save the index to disk
        faiss_index.save(self.index_filename(self.subset_dirs[subset_name]))
        # Save the index map to disk
        with open(self.index_map_filename(self.subset_dirs[subset_name]), 'w') as f:
            json.dump(cur_paths, f)


    def load_index(self, subset_path):
        index_file = self.index_filename(subset_path)
        map_file = self.index_map_filename(subset_path)
        if os.path.exists(index_file) and os.path.exists(map_file):
            faiss_index = FAISSIndexer(dimension=384)
            faiss_index.load(index_file)
            with open(map_file, 'r') as f:
                cur_paths = json.load(f)
            subset_name = os.path.basename(subset_path)
            self.img_index[subset_name] = faiss_index
            self.map_idx_to_path[subset_name] = cur_paths
            self.map_idx_to_name[subset_name] = {
                i: os.path.basename(p).split('.')[0] for i, p in enumerate(cur_paths)}
            self.map_name_to_idx[subset_name] = {
                k: v for v, k in self.map_idx_to_name[subset_name].items()}
            return True
        else:
            return False
        
    def query(self, subset, query_img_path, k=3):
        """
        Query the index with an image path.
        
        :param subset: Subset name to query.
        :param query_img_path: Path of the query image.
        :param k: Number of top k similar images to retrieve.
        :return: List of top_k similar image paths.
        """
        if subset not in self.img_index:
            raise ValueError(f"Subset {subset} not found in index database.")
        if not os.path.exists(query_img_path):
            raise ValueError(f"Query image {query_img_path} does not exist.")
        
        subset_name = os.path.basename(subset)
        image = Image.open(query_img_path).convert('RGB')
        image = self.src_preprocess(image).unsqueeze(0).cuda()
        emb_dict = self.generate_embedding(image)
        cls_token = emb_dict['x_norm_clstoken']  # (1, C)
        faiss_index = self.img_index[subset_name]
        if faiss_index is None or len(self.map_idx_to_path[subset_name]) == 0:
            raise ValueError(f"No index found for dataset {subset_name}. Please generate the index first.")
        distances, indices = faiss_index.search(cls_token, k=k)
        query_img_paths = [self.map_idx_to_path[subset_name][idx] for idx in indices[0]]
        return query_img_paths



class SYNTRAinf:
    def __init__(self, model_name, backbone, dora_rank, mask_decoder, checkpoint, uni_transform, **kwargs):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build transforms
        self.uni_transform = TransformCompose(uni_transform) 

        # build model
        checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = build_segmentation_model(
            model_name=model_name,
            backbone=backbone,
            dora_rank=dora_rank,
            mask_decoder=mask_decoder,
        )
        self.model.load_trainable_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

    def prepare_data(self, source_images, source_labels, target_image):
        # transform source images and labels
        source_images, source_labels = self.uni_transform(source_images, source_labels)
        source_images = torch.stack(source_images, dim=0)  # (T-1, 3, H, W)
        source_labels = torch.cat(source_labels, dim=0)    # (T-1, H/4, W/4)
        source_labels = source_labels[:, ::4, ::4]  # downsample labels by 4

        # convert source labels to multi-notion format
        selected_cls_ids = [x for x in source_labels.unique().tolist() if x != 255 and x != 0]
        multi_notions_src_lbls = []
        for cls_id in selected_cls_ids:
            multi_notions_src_lbls.append((source_labels == cls_id))
        multi_notions_src_lbls = torch.stack(multi_notions_src_lbls, dim=1).to(torch.int16)  # (T-1, N, H, W)

        # transform target image
        target_image, _ = self.uni_transform(target_image, None)
        target_image = target_image  # (3, H, W)

        # load to gpu
        img_src = source_images.unsqueeze(0).to(self.device)  # (1, T-1, 3, H, W)
        lbl_src = multi_notions_src_lbls.unsqueeze(0).to(self.device)  # (1, T-1, N, H/4, W/4)
        img_tgt = target_image.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        return img_src, lbl_src, img_tgt, selected_cls_ids

    def segment_target(self, source_images, source_labels, target_image):
        # prepare data
        img_src, lbl_src, img_tgt, selected_cls_ids = self.prepare_data(source_images, source_labels, target_image)

        # forward pass
        with torch.no_grad():
            pred = self.model(img_src, lbl_src, img_tgt).sigmoid().squeeze(0).squeeze(1)  # (N, H, W)

        return pred, selected_cls_ids
    

def cal_iou(pred_mask, target_mask, cls_id):
    pred_binary_mask = (pred_mask == cls_id).astype(np.uint8)
    target_binary_mask = (target_mask == cls_id).astype(np.uint8)
    intersection = np.logical_and(pred_binary_mask, target_binary_mask).sum()
    union = np.logical_or(pred_binary_mask, target_binary_mask).sum()
    iou = intersection / union if union != 0 else 1.0
    return iou

def cal_acc(pred_mask, target_mask):
    correct = (pred_mask == target_mask).sum()
    total = target_mask.size
    acc = correct / total
    return acc


def load_config(config_name: str):
    # Initialize Hydra and compose the configuration
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module="syntra.configs", version_base=None)
    cfg = compose(config_name=config_name)
    return cfg


def inference(args, cfg):

    # build source indexing database
    # build model
    syntra = SYNTRAinf(args)
    # build 


def test(args, cfg):
    # ensure result directory
    if args.output_dir is None: 
        args.output_dir = os.path.dirname(args.checkpoint)

    result_dir = os.path.join(args.output_dir, 'inf_results')
    results = {}
    os.makedirs(result_dir, exist_ok=True)
    # build source indexing database
    index_db = SourceIndexDatabase(args.data_root, 
                                   source_sample_list_file=args.indexing_file)
    # build model
    syntra = SYNTRAinf(checkpoint=args.checkpoint, **cfg.TRAIN)

    # process subsets
    for subset_name, subset_dir in index_db.subset_dirs.items():
        print(f"Processing subset: {subset_name}")

        # parse target sample list
        target_sample_list_file = os.path.join(subset_dir, 'test_samples.txt')
        with open(target_sample_list_file, 'r') as f:
            target_filenames = [line.strip().split(".")[-1] for line in f.readlines()]
        # load color map
        color_map_file = os.path.join(subset_dir, 'color_map.json')
        with open(color_map_file, 'r') as f:
            color_map = {k: v for k, v in json.load(f).items() if k not in ['background', 'nonlabeled']}
        cls_id_to_name = {i+1: name for i, name in enumerate(color_map.keys())}
        results[subset_name] = {"ious": {k: [] for k in color_map.keys()}, "accs": []}

        for target_filename in target_filenames:
            # load target image and label
            target_img_path = os.path.join(subset_dir, 'imgs', target_filename + '.png')
            target_img = Image.open(target_img_path).convert('RGB')
            target_lbl_path = target_img_path.replace('/imgs/', '/lbls/')
            target_lbl = Image.open(target_lbl_path)
            source_image_paths = index_db.query(subset_name, target_img_path, k=args.num_src)

            # load source images and labels
            source_images = []
            source_labels = []
            for src_path in source_image_paths:
                src_img = Image.open(src_path).convert('RGB')
                source_images.append(src_img)
                # load corresponding label
                src_lbl_path = src_path.replace('/imgs/', '/lbls/')
                src_lbl = Image.open(src_lbl_path)
                source_labels.append(src_lbl)
                
            # perform segmentation
            pred_masks, selected_cls_ids = syntra.segment_target(source_images, source_labels, target_img)
            pred_segmentation = pred_masks.argmax(dim=0) + 1  # add 1 to match class ids
            pred_segmentation[(pred_masks < 0.5).all(dim=0)] = 0  # set to background if below threshold
            pred_segmentation = pred_segmentation.cpu().numpy()

            # evaluate
            target_lbl = np.array(target_lbl)
            for cls_id, cls_name in cls_id_to_name.items():
                if cls_id not in selected_cls_ids:
                    pred_binary_mask = np.zeros_like(target_lbl, dtype=np.uint8)
                    pred_segmentation[pred_segmentation == selected_cls_ids.index(cls_id)] = cls_id
                else:
                    pred_binary_mask = (pred_masks[selected_cls_ids.index(cls_id)].cpu().numpy() >= 0.5).astype(np.uint8)
                iou = cal_iou(pred_binary_mask, target_lbl, cls_id)
                results[subset_name]["ious"][cls_name].append(iou)
                

            acc = cal_acc(pred_segmentation, target_lbl)
            results[subset_name]["accs"].append(acc)
            
            # save predicted segmentation
            pred_seg_pil = Image.fromarray(pred_segmentation.astype(np.uint8))
            pred_seg_pil.putpalette([0, 0, 0] + sum(color_map.values(), []))
            pred_seg_pil.save(os.path.join(result_dir, f"{subset_name}.{target_filename}.png"))

                

if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="sub-commands: inference or test")
    
    # sub-commands for test
    parser_test = subparsers.add_parser("test", help="inference")
    parser_test.add_argument("-d", "--data_root", type=str, help="path to dataset root directory")
    parser_test.add_argument("-i", "--indexing-file", type=str, help="filename that list all source samples for indexing")
    parser_test.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_test.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_test.add_argument("-c", "--config", type=str, help="config file name")
    parser_test.add_argument("-n", "--num-src", type=int, default=3, help="number of source images")

    # sub-commands for inference
    parser_inference = subparsers.add_parser("inference", help="inference")
    parser_inference.add_argument("-s", "--source-data-dir", type=str, help="path to source dataset root directory")
    parser_inference.add_argument("-t", "--target-data-dir", type=str, help="path to target dataset root directory")
    parser_inference.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser_inference.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser_inference.add_argument("-c", "--config", type=str, help="config file name")
    parser_inference.add_argument("-n", "--num-src", type=int, default=3, help="number of source images")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "inference":
        inference(args, cfg)
    elif args.command == "test":
        test(args, cfg)
    else:
        print("Please specify a valid command: inference or test")