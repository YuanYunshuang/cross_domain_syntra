
import copy
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse



def construct_color_palette(color_map):
        """
        Constructs a color palette list from the color map.
        Parameters:
            color_map (dict): A dictionary mapping class names to RGB color lists.
        Returns:
            list: A flat list representing the color palette.
        """
        palette = [0, 0, 0]  # background color
        for cls_name in color_map.keys():
            palette.extend(color_map[cls_name])
        # pad the palette to have 256 colors
        palette.extend([0, 0, 0] * (256 - len(color_map) - 1))
        # set the last color to white as nonlabled
        palette[-3:] = [255, 255, 255]
        return palette


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / (np.sum(np.exp(x), axis=0) + 1e-6)


def iou_score(y_true, y_pred, ignore_mask=None, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) between the ground truth and predicted masks.

    Parameters:
    y_true (ndarray): Ground truth binary mask.
    y_pred (ndarray): Predicted binary mask.
    ignore_mask (ndarray, optional): Binary mask to ignore certain regions. Defaults to None.
    smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    float: IoU Score.
    """
    if ignore_mask is not None:
        y_true = y_true * (1 - ignore_mask)
        y_pred = y_pred * (1 - ignore_mask)
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def accuracy_score(y_true, y_pred, ignore_mask=None):
    """
    Compute the accuracy between the ground truth and predicted masks.

    Parameters:
    y_true (ndarray): Ground truth binary mask.
    y_pred (ndarray): Predicted binary mask.
    ignore_mask (ndarray, optional): Binary mask to ignore certain regions. Defaults to None.

    Returns:
    float: Accuracy Score.
    """
    if ignore_mask is not None:
        valid_mask = 1 - ignore_mask
        correct = ((y_true == y_pred) * valid_mask).sum()
        total = valid_mask.sum()
    else:
        correct = (y_true == y_pred).sum()
        total = y_true.size
    accuracy = correct / (total + 1e-6)
    return accuracy

def eval(result_folder, data_folder):
    """
    Evaluate segmentation results using Dice Score.
    Parameters:
        result_folder (str): Path to the folder containing N folders, each with predicted masks for one cls.
            Note that the file names should match those in ground truth folder.
        data_folder (str): syntra dataset folder containing folder 'lbls' and 'color_map.json'.
    """
    subsets = os.listdir(result_folder)
    results = {k: {'acc': [], 'iou': {}} for k in ['hameln', 'bayern', 'siegfried']}
    for subset in subsets:
        print(f"Processing subset: {subset}...")
        dataset_name = subset.split('.')[0]
        cur_data_folder = os.path.join(data_folder, subset)
        with open(os.path.join(cur_data_folder, 'color_map.json'), 'r') as f:
            color_map = json.load(f)
        color_palette = construct_color_palette(color_map)

        classes = [x for x in color_map.keys() if x not in ['nonlabeled', 'background']]
        res_classes = sorted(os.listdir(os.path.join(result_folder, subset)))
        results[dataset_name]['iou'] = {cls_name: [] for cls_name in classes}
        gt_folder = os.path.join(cur_data_folder, 'lbls')
        img_folder = os.path.join(cur_data_folder, 'imgs')
        vis_folder = os.path.join(os.path.dirname(result_folder), 'merged_vis', subset)
        os.makedirs(vis_folder, exist_ok=True)

        test_mask_files = sorted(os.listdir(os.path.join(result_folder, subset, res_classes[0])))

        for file_name in test_mask_files:
            gt_mask = np.array(Image.open(os.path.join(gt_folder, file_name)))
            input_img = Image.open(os.path.join(img_folder, file_name))
            pred_mask = []
            for cls_name in classes:
                if cls_name not in res_classes:
                    pred_mask.append(np.zeros_like(gt_mask))
                    continue
                pred_mask_path = os.path.join(result_folder, subset, cls_name, file_name)
                pred_mask.append(np.array(Image.open(pred_mask_path)))

            pred_mask = np.stack(pred_mask, axis=0) / 255  # (N, H, W)
            pred_prob = softmax(pred_mask)  # (N, H, W)
            pred_score, pred_cls = pred_prob.max(axis=0), pred_prob.argmax(axis=0)  # (H, W), (H, W)
            pred_cls += 1  # to match gt mask which starts from 1
            pred_cls[pred_mask.max(axis=0) < 0.5] = 0  # background
        
            acc = accuracy_score(gt_mask, pred_cls)
            results[dataset_name]['acc'].append(acc)
            for i, cls_name in enumerate(classes):
                gt_cls_mask = (gt_mask == (i + 1)).astype(np.uint8)
                pred_cls_mask = (pred_cls == (i + 1)).astype(np.uint8)
                iou = iou_score(gt_cls_mask, pred_cls_mask)
                results[dataset_name]['iou'][cls_name].append(iou)
            
            # visualize merged predictions
            vis_lbl = np.concatenate([gt_mask, pred_cls], axis=1).astype(np.uint8)
            vis_lbl = Image.fromarray(vis_lbl).convert("P")
            vis_lbl.putpalette(color_palette)
            vis_img_lbl = Image.new('RGB', (input_img.width + vis_lbl.width, input_img.height))
            vis_img_lbl.paste(input_img, (0, 0))
            vis_img_lbl.paste(vis_lbl, (input_img.width, 0))
            vis_img_lbl.save(os.path.join(vis_folder, file_name))
        
    res_str = ""
    mean_ious = []
    for dataset, res in results.items():
        mean_acc = np.mean(res['acc'])
        for cls_name, ious in res['iou'].items():
            mean_iou = np.mean(ious)
            mean_ious.append(mean_iou)
            res_str += f"{dataset} - Class {cls_name}: IoU: {mean_iou:.4f}\n"
        
        res_str += f"{dataset} - Mean IoU: {mean_iou:.4f}\n"
        res_str += f"{dataset} - Mean Accuracy: {mean_acc:.4f}\n"
        res_str += "\n"
    overall_mean_iou = np.mean(mean_ious)
    res_str += f"Overall Mean IoU: {overall_mean_iou:.4f}\n"
    
    print(res_str)
    with open(os.path.join(os.path.dirname(result_folder), 'hismap_eval_results.txt'), 'w') as f:
        f.write(res_str)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, required=True, help='Path to the result folder.')
    parser.add_argument('--data-folder', type=str, required=True, help='Path to the data folder.')
    args = parser.parse_args()
    eval(args.result_folder, args.data_folder)
    # result_folder = "/koko/SYNTRA_logs/syntra_hismap_n1_s100_v2/hismap"
    # data_folder = "/koko/datasets/syntra/hismap"
    # eval(result_folder, data_folder)

