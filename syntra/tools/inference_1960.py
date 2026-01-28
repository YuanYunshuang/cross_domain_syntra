# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import os
import sys
import random
import argparse
from tifffile import imread
from tqdm import tqdm
from glob import glob
import json

    
import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2


from syntra.modeling.dataset.trainsform import TransformCompose
from syntra.modeling import build_segmentation_model
from data_tools.indexing.faiss_utils import FAISSIndexer

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

color_map = {
  "forest": [34, 139, 34],          
  "grass": [124, 252, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
}

cls_name_to_id = {name: idx+1 for idx, name in enumerate(color_map.keys())}
palette = [0, 0, 0] + sum(color_map.values(), [])  # adding background color


def set_seeds(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path



class BaseSourceDataset(Dataset):
    def __init__(self, data_root, subsets, nshot="100", ext='.png', normalize=True):
        self.ext = ext
        self.normalize = normalize
        self.data_root = data_root 
        self.subsets = subsets
        self.nshot = nshot

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
        if len(self.subsets) > 0:
            self.parse_sample_list()
    
    @property
    def active_class(self):
        if not hasattr(self, 'active_cls'):
            raise ValueError("Active class not set. Please call set_cls(cls_name) to set the active class.")
        return self.active_cls

    def parse_sample_list(self):
        self.filenames = {}
        subset_files = []
        for subset in self.subsets:
            subset_files.extend(glob(os.path.join(self.data_root, subset, f'cls_selection_{self.nshot}.json')))
        for subset_file in subset_files:
            with open(subset_file, 'r') as f:
                cls_files = json.load(f)
            for cls_name, files in cls_files.items():
                if cls_name not in self.filenames:
                    self.filenames[cls_name] = []
                self.filenames[cls_name].extend(files)
    
    def set_cls(self, cls_name):
        if cls_name not in self.filenames:
            raise ValueError(f"Class {cls_name} not found in dataset.")
        self.active_cls = cls_name

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[self.active_class][idx]
        subset, filename = filename.rsplit('.', 1)
        full_path = os.path.join(self.data_root, subset, 'imgs', filename) + self.ext

        image = Image.open(full_path).convert('RGB')
        image = self.preprocess(image)  
        return full_path, image
    
    def collate_fn(self, batch):
        paths, images = zip(*batch)
        images = torch.stack(images, dim=0)
        return paths, images


def build_source_dataloader(data_root, subsets, nshot="100", batch_size=32, shuffle=False, num_workers=4, ext='.png', normalize=True):
    dataset = BaseSourceDataset(data_root, subsets, nshot, ext, normalize=normalize)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return dataloader
    

class SourceIndexDatabase:
    def __init__(self, 
                 root_dir, 
                 subsets,
                 nshot=100, # 100, 50, 10
                 index_dim=384, 
                 cache_original_vec=True):
        """
        Docstring for __init__
        
        :param self: Instance of the IndexDatabase class
        :param root_dir: Root directory containing the data subsets
        :param subsets: List of subset directory names to index
        :param nshot: Number of shots for few-shot indexing
        :param index_dim: Dimension of the index vectors
        :param cache_original_vec: Cache the original vectors while indexing or not
        """
        self.root_dir = root_dir
        self.subsets = subsets
        self.nshot = nshot
        self.index_dim = index_dim
        self.cache_original_vec = cache_original_vec
        self.img_index = {}
        self.encoded_vectors = {}
        self.subset_dirs = {}
        self.map_idx_to_path = {}
        self.map_idx_to_name = {}
        self.map_name_to_idx = {}

        self.src_preprocess = BaseSourceDataset("", []).preprocess
        self.cls_names = [x for x in color_map.keys()]

        self.build_img_encoder()
        self.build_index_database()

    @property
    def datasets(self):
        return list(self.img_index.keys())
    
    def build_img_encoder(self):
        self.img_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', source='github')
        self.img_encoder.cuda().eval()  # Set the model to evaluation mode
    
    def generate_embedding(self, input_tensor):
        # Generate embeddings
        with torch.no_grad():  # Disable gradient computation
            embeddings = self.img_encoder.forward_features(input_tensor)

        return embeddings
    
    def indexing(self, dataloader, cls_name):
        dataloader.dataset.set_cls(cls_name)
        faiss_index = FAISSIndexer(dimension=384)  # DINOv2 Vit-S/14 embedding dimension
        cur_paths = []
        cur_img_vectors = []
        
        for paths, imgs in dataloader:
            imgs = imgs.cuda()
            emb_dict = self.generate_embedding(imgs)
            token = emb_dict['x_norm_clstoken']  # (B, C)
            cur_paths.extend(paths)
            faiss_index.add(token)
            if self.cache_original_vec:
                cur_img_vectors.append(token)
        
        # self.img_index[subset_name] = faiss_index
        # if self.cache_original_vec:
        #     self.encoded_vectors[subset_name] = torch.cat(cur_img_vectors, dim=0)
        # self.map_idx_to_path[subset_name] = cur_paths
        # self.map_idx_to_name[subset_name] = {
        #     i: os.path.basename(p).split('.')[0] for i, p in enumerate(cur_paths)}
        # self.map_name_to_idx[subset_name] = {
        #     k: v for v, k in self.map_idx_to_name[subset_name].items()}
        return faiss_index, cur_paths, torch.cat(cur_img_vectors, dim=0) 

    def build_index_database(self):
        dataloader = build_source_dataloader(data_root=self.root_dir, 
                                             subsets=self.subsets, 
                                             nshot=self.nshot)
        for cls_name in self.cls_names:
            print(f"Indexing class: {cls_name}")
            faiss_index, cur_paths, cur_img_vectors = self.indexing(dataloader, cls_name)
            self.img_index[cls_name] = faiss_index
            if self.cache_original_vec:
                self.encoded_vectors[cls_name] = cur_img_vectors
            self.map_idx_to_path[cls_name] = cur_paths
            self.map_idx_to_name[cls_name] = {
                i: os.path.basename(p).split('.')[0] for i, p in enumerate(cur_paths)}

    
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
        
    def query(self, cls_name, query_img, k=3):
        """
        Query the index with an image path.
        
        :param subset: Subset name to query.
        :param query_img: numpy array or PIL Image of the query image.
        :param k: Number of top k similar images to retrieve.
        :return: List of top_k similar image paths.
        """
        if cls_name not in self.img_index:
            raise ValueError(f"Dataset {cls_name} not found in the index database.")
        if isinstance(query_img, np.ndarray):
            query_img = Image.fromarray(query_img)
        image = self.src_preprocess(query_img).unsqueeze(0).cuda()
        emb_dict = self.generate_embedding(image)
        cls_token = emb_dict['x_norm_clstoken']  # (1, C)
        faiss_index = self.img_index[cls_name]
        _, indices = faiss_index.search(cls_token, k=k)
        query_img_paths = [self.map_idx_to_path[cls_name][idx] for idx in indices[0]]
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
        multi_notions_src_lbls = source_labels.unsqueeze(1).to(torch.int16)  # (T-1, 1, H, W)

        # transform target image
        if isinstance(target_image, np.ndarray):
            target_image = Image.fromarray(target_image)
        target_image, _ = self.uni_transform(target_image, None)
        target_image = target_image  # (3, H, W)

        # load to gpu
        img_src = source_images.unsqueeze(0).to(self.device)  # (1, T-1, 3, H, W)
        lbl_src = multi_notions_src_lbls.unsqueeze(0).to(self.device)  # (1, T-1, N, H/4, W/4)
        img_tgt = target_image.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        return img_src, lbl_src, img_tgt

    def segment_target(self, source_images, source_labels, target_image):
        """
        Segment the target image given source images and labels.
        :param source_images: list of PIL Images of source images
        :param source_labels: list of PIL Images of source labels
        :param target_image: numpy array or PIL Image of target image
        :return: predicted mask of the target image
        """
        # prepare data
        img_src, lbl_src, img_tgt = self.prepare_data(source_images, source_labels, target_image)

        # forward pass
        with torch.no_grad():
            pred = self.model(img_src, lbl_src, img_tgt).sigmoid().squeeze(0).squeeze(1)  # (N, H, W)

        return pred
    

def cal_iou(pred_mask, target_mask):
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
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
    result_dir = os.path.join(args.output_dir, 'inf_results')
    results = {}
    os.makedirs(result_dir, exist_ok=True)
    # build source indexing database
    index_db = SourceIndexDatabase(args.data_root,
                                   subsets=args.source_keys.split(','),
                                   nshot=args.nshot)
    # build model
    syntra = SYNTRAinf(checkpoint=args.checkpoint, **cfg.TRAIN)

    # load map and label
    map_sheet = imread(os.path.join(args.target_map_dir, '1960.tif'))
    label_sheet = Image.open(os.path.join(args.target_map_dir, '1960_lbl.png'))
    label_sheet = np.array(label_sheet)

    # prepare for evaluation
    sliding_window_size = 384
    overlap_size = 128
    height, width, _ = map_sheet.shape
    if args.visualize:
        vis_win = cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)

    # prediction
    for cls_name, cls_id in cls_name_to_id.items():
        print(f"Processing class: {cls_name}")
        label_pred = np.zeros_like(label_sheet, dtype=float)
        label_gt = (label_sheet == cls_id).astype(np.uint8) * cls_id
        for x in range(0, height, sliding_window_size - overlap_size):
            for y in range(0, width, sliding_window_size - overlap_size):
                x_end = min(x + sliding_window_size, height)
                y_end = min(y + sliding_window_size, width)
                x_start = max(0, x_end - sliding_window_size)
                y_start = max(0, y_end - sliding_window_size)

                target_img = map_sheet[x_start:x_end, y_start:y_end, :]
                target_lbl = label_gt[x_start:x_end, y_start:y_end]

                source_image_paths = index_db.query(cls_name, target_img, k=args.num_src)

                # load source images and labels
                source_images = []
                source_labels = []
                for src_path in source_image_paths:
                    src_img = Image.open(src_path).convert('RGB')
                    source_images.append(src_img)
                    # load corresponding label
                    src_lbl_path = src_path.replace('/imgs/', '/lbls/')
                    src_lbl = Image.open(src_lbl_path)
                    src_lbl = np.array(src_lbl) == cls_id
                    source_labels.append(src_lbl)

                # perform segmentation
                pred_mask = syntra.segment_target(source_images, source_labels, target_img).squeeze(0).cpu().numpy()
                # merge with previous predictions
                pred_mask = np.maximum(label_pred[x_start:x_end, y_start:y_end], pred_mask)
                label_pred[x_start:x_end, y_start:y_end] = pred_mask

                # visualize
                if args.visualize:
                    vis_img = Image.new('RGB', (sliding_window_size * (args.num_src + 2), sliding_window_size))
                    vis_lbl = Image.new('P', (sliding_window_size * (args.num_src + 2), sliding_window_size))
                    vis_lbl.putpalette(palette)
                    for i, src_img in enumerate(source_images):
                        src_lbl = Image.fromarray(source_labels[i].astype(np.uint8) * cls_id)
                        vis_img.paste(src_img, (i * sliding_window_size, 0))
                        vis_lbl.paste(src_lbl, (i * sliding_window_size, 0))
                    target_img = Image.fromarray(target_img)
                    vis_img.paste(target_img, (args.num_src * target_img.width, 0))
                    vis_lbl.paste(Image.fromarray(target_lbl.astype(np.uint8)), (args.num_src * target_img.width, 0))
                    vis_lbl.paste(Image.fromarray((pred_mask > 0.5).astype(np.uint8) * cls_id), ((args.num_src + 1) * target_img.width, 0))

                    vis = np.concatenate([np.array(vis_img), np.array(vis_lbl.convert('RGB'))], axis=0)
                    cv2.imshow("Visualization", vis[:, :, ::-1])
                    # add tile info
                    cv2.setWindowTitle("Visualization", f"{cls_name}: {x}, {y}")
                    cv2.waitKey(0)

        
        Image.fromarray((label_pred *255).astype(np.uint8)).save(os.path.join(result_dir, f"1960_pred_{cls_name}.png"))

        # evaluate
        pred_mask = label_pred >= 0.5
        gt_mask = label_gt > 0
        iou = cal_iou(pred_mask, gt_mask)
        acc = cal_acc(pred_mask, gt_mask)
        results[cls_name] = {"iou": iou, "acc": acc}
        print(f"Class: {cls_name}, IoU: {iou*100:.2f}, Acc: {acc*100:.2f}")




if __name__ == "__main__":
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", type=str, help="path to dataset root directory")
    parser.add_argument("-s", "--source-keys", type=str, help="source keys for filtering subsets relative paths for source indexing, split by comma")
    parser.add_argument("-t", "--target-map-dir", type=str, help="target map directory, it contains map_name.tif and map_name_lbl.tif")
    parser.add_argument("-w", "--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("-o", "--output-dir", type=str, help="path to output directory")
    parser.add_argument("-c", "--config", type=str, help="config file name")
    parser.add_argument("-n", "--num-src", type=int, default=3, help="number of source images")
    parser.add_argument("--nshot", type=int, default=100, help="number of shots for source indexing, options: 100, 50, 10")
    parser.add_argument("-vis", "--visualize", action='store_true', help="whether to visualize the results during inference")

    args = parser.parse_args()
    cfg = load_config(args.config)

    inference(args, cfg)
