# Decription: Generate src-tgt pairing info to json file. Given a target image, find its k nearest source images 
# Author: Yunshuang Yuan
# Date: 2025-10-01
# All rights reserved.

import os
import json

from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from tifffile import imread as tiff_imread

from .indexdb import IndexDatabase, DinoV2Encoder, BaseDataloaders


def generate_src_tgt_intra_cls(data_root, tgt_info_file, src_info_file=None, k=8, folder_list=None, fg_ratio=0.05, visualize=False):
    if src_info_file is None:
        src_info_file = tgt_info_file
    if folder_list is None:
        folder_list = [os.path.dirname(x) for x in glob(os.path.join(data_root, "**/color_map.json"), recursive=True)]
    img_encoder = DinoV2Encoder()
    for dataset_full_path in folder_list:
        if not os.path.isdir(dataset_full_path):
            print(f"Warning: {dataset_full_path} is not a valid directory. Skip.")
            continue
        
        dataset = os.path.basename(dataset_full_path)
        with open(os.path.join(dataset_full_path, 'color_map.json'), 'r') as f:
            class_names = json.load(f).keys()
        with open(os.path.join(dataset_full_path, f'{dataset}.json'), 'r') as f:
            data_info = json.load(f)
        
        tgt_samples = load_samples(os.path.join(dataset_full_path, tgt_info_file), remove_repeat=False)
        src_samples = load_samples(os.path.join(dataset_full_path, src_info_file), remove_repeat=True)
        src_cls_samples = parse_class_samples(data_info, src_samples, class_names)
        
        pairing_info = {}
        for cls_name in class_names:
            if len(src_cls_samples[cls_name]) == 0:
                print(f"Warning: No source samples found for class {cls_name} in dataset {dataset}. Skip.")
                continue
            print(f"Processing dataset: {dataset}, class: {cls_name}")
            pairing_info[cls_name] = {}
            cur_src_samples = src_cls_samples[cls_name]
                    # build index db for current source samples
            ext = "." + os.listdir(os.path.join(dataset_full_path, 'imgs'))[0].rsplit('.', 1)[-1]
            cur_data_root = os.path.dirname(dataset_full_path)
            map_dataloader = BaseDataloaders(cur_data_root, cur_src_samples, batch_size=4, ext=ext)
            index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)
    
            for sample in tqdm(tgt_samples):
                sample_name = sample.rsplit('.', 1)[1] 
                sample_img_path = os.path.join(dataset_full_path, 'imgs', f"{sample_name}{ext}")
                distances, indices =  index_db.external_search(
                    dataset, sample_img_path, k=k) 
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
        src_mode = os.path.splitext(src_info_file)[0].replace('selected_', '').replace('_samples', '')
        tgt_mode = os.path.splitext(tgt_info_file)[0].replace('selected_', '').replace('_samples', '')
        save_path = os.path.join(dataset_full_path, f'pairing_{src_mode}_{tgt_mode}_intracls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)
        
        if visualize:
            visualize_generateion(dataset, pairing_info, mode='intra', n_vis = 5, ext=ext)


def load_samples(data_info_file, remove_repeat=True, keep_cls_structure=False):
    if '.json' in data_info_file:
        with open(data_info_file, 'r') as f:
            data_info = json.load(f)
        if keep_cls_structure:
            return data_info
        all_samples = []
        for cls_samples in data_info.values():
            all_samples += cls_samples
        
    else:
        with open(data_info_file, 'r') as f:
            all_samples = f.read().splitlines()
    if remove_repeat:
        all_samples = list(set(all_samples))
    return all_samples


def parse_class_samples(data_info, samples, class_names, fg_ratio=0.05):
    class_samples = {x: [] for x in class_names}
    for sample in samples:
        for cls, ratio in data_info[sample].items():
            if cls in ['background', 'nonlabeled']:
                continue
            if ratio > fg_ratio:
                class_samples[cls].append(sample)
    return class_samples


def generate_src_tgt_inter_cls(data_root, tgt_info_file, src_info_file=None, k=8, folder_list=None, visualize=False):
    img_encoder = DinoV2Encoder()
    if src_info_file is None:
        src_info_file = tgt_info_file
    if folder_list is None:
        folder_list = os.listdir(data_root)
    for dataset in folder_list:
        if not os.path.isdir(os.path.join(data_root, dataset)):
            continue

        cur_tgt_info_file = os.path.join(data_root, dataset, tgt_info_file)
        cur_src_info_file = os.path.join(data_root, dataset, src_info_file)

        tgt_samples = load_samples(cur_tgt_info_file, remove_repeat=False)
        src_samples_no_repeat = load_samples(cur_src_info_file, remove_repeat=True)
        
        pairing_info = {}
    
        # build index db for all samples
        ext = "." + os.listdir(os.path.join(data_root, dataset, 'imgs'))[0].rsplit('.', 1)[-1]
        map_dataloader = BaseDataloaders(data_root, src_samples_no_repeat, batch_size=4, ext=ext)
        index_db = IndexDatabase(map_dataloader=map_dataloader, img_encoder=img_encoder, merge_patch_token=False)

        for sample in tgt_samples:
            sample_name = sample.rsplit('.', 1)[1] 
            sample_img_path = os.path.join(data_root, dataset, 'imgs', f"{sample_name}{ext}")
            distances, indices =  index_db.external_search(
                dataset, sample_img_path, k=k) 
            # remove self from the pairing list
            mask = distances>1e-3
            distances = distances[mask]
            indices = indices[mask]
            if len(indices) < k - 1:
                print(f"Warning: sample {sample} has less than {k-1} neighbors. Final selected neighbors: {len(indices)}")
            # update pairing info
            pairing_info[sample] = {
                'src_samples': [src_samples_no_repeat[i] for i in indices],
                'distances': distances.tolist()
            }
        del index_db
        del map_dataloader

        # save pairing info
        src_mode = os.path.splitext(src_info_file)[0].replace('selected_', '').replace('_samples', '')
        tgt_mode = os.path.splitext(tgt_info_file)[0].replace('selected_', '').replace('_samples', '')
        save_path = os.path.join(data_root, dataset, f'pairing_{src_mode}_{tgt_mode}_intercls.json')
        with open(save_path, 'w') as f:
            json.dump(pairing_info, f, indent=4)
        
        if visualize:
            visualize_generateion(dataset, pairing_info, mode='inter', n_vis = 5, ext=ext)


def visualize_one_pairing(tgt_sample, src_samples, img_dir, ext='.png'):
    tgt_name = tgt_sample.rsplit('.', 2)[-1]
    src_names = [s.rsplit('.', 2)[-1] for s in src_samples]
    n_src = len(src_names)
    fig, ax = plt.subplots(1, n_src+1, figsize=(4*(n_src+1), 4))
    fig.suptitle(f"Target: {tgt}", fontsize=16)
    if ext == '.png':
        tgt_img = Image.open(os.path.join(img_dir, f"{tgt_name}{ext}")).convert("RGB")
    elif ext == '.tiff':
        tgt_img = tiff_imread(os.path.join(img_dir, f"{tgt_name}{ext}"))
        # tgt_img = (np.array(tgt_img) * 255).astype(np.uint8)
    ax[0].imshow(tgt_img)
    ax[0].set_title("Target")
    ax[0].axis('off')
    for i, src_name in enumerate(src_names):
        if ext == '.png':
            src_img = Image.open(os.path.join(img_dir, f"{src_name}{ext}")).convert("RGB")
        elif ext == '.tiff':
            src_img = tiff_imread(os.path.join(img_dir, f"{src_name}{ext}"))
            # src_img = (np.array(src_name) * 255).astype(np.uint8)
        
        ax[i+1].imshow(src_img)
        ax[i+1].set_title(f"Source {i+1}")
        ax[i+1].axis('off')
    plt.show()


def visualize_generateion(dataset, pairing_info, mode='inter', n_vis = 5, ext='.png'):

        img_dir = os.path.join(data_root, dataset, 'imgs')
        # select a few samples to visualize 
        if mode == 'intra':
            # select 5 samples per class
            for cls, samples in pairing_info.items():
                vis_keys = np.random.choice(list(samples.keys()), size=n_vis, replace=False).tolist()
                for tgt in vis_keys:
                    info = samples[tgt]
                    visualize_one_pairing(tgt, info['src_samples'], img_dir, ext=ext)
        else:
            vis_keys = np.random.choice(list(pairing_info.keys()), size=n_vis, replace=False).tolist()
            for tgt in vis_keys:
                info = pairing_info[tgt]
                visualize_one_pairing(tgt, info['src_samples'], img_dir, ext=ext)


if __name__=="__main__":
    data_root = "/koko/datasets/syntra/forest_monitoring"
    folder_list_all = glob(os.path.join(data_root, "**/color_map.json"), recursive=True)
    folder_list_all = [os.path.dirname(x) for x in folder_list_all]
    folder_list_with_selection = glob(os.path.join(data_root, "**/train_100shot.txt"), recursive=True)
    folder_list_with_selection = [os.path.dirname(x) for x in folder_list_with_selection]
    folder_list_no_selection = list(set(folder_list_all) - set(folder_list_with_selection))

    
    for tgt_info_file in ['train_samples.txt', 'test_samples.txt']:
        generate_src_tgt_intra_cls(data_root,
                                tgt_info_file=tgt_info_file,
                                src_info_file='train_samples.txt',
                                folder_list=folder_list_no_selection,
                                k=8, 
                                visualize=True)
    
    for tgt_info_file in ['train_100shot.txt', 'test_samples.txt']:
        generate_src_tgt_intra_cls(data_root,
                                tgt_info_file=tgt_info_file,
                                src_info_file='train_100shot.txt',
                                folder_list=folder_list_with_selection,
                                k=8, 
                                visualize=True)