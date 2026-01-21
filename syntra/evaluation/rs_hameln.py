
import copy
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm



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
    res_classes = sorted(os.listdir(result_folder))

    
    with open(os.path.join(data_folder, 'color_map.json'), 'r') as f:
        color_map = json.load(f)
    color_palette = construct_color_palette(color_map)

    classes = list(color_map.keys())
    scores = {cls_name: [] for cls_name in classes}
    overall_acc = []
    gt_folder = os.path.join(data_folder, 'lbls')
    vis_folder = os.path.join(os.path.dirname(result_folder), 'merged_vis')
    os.makedirs(vis_folder, exist_ok=True)

    test_mask_files = sorted(os.listdir(os.path.join(result_folder, classes[0])))
    merged_res = {}
    for file_name in tqdm(test_mask_files):
        patch_id, y, x = file_name.split('.')[0].rsplit('_', 2)
        x, y = int(x), int(y)
        if patch_id not in merged_res:
            merged_res[patch_id] = {
                'score': np.zeros((512, 512)), 
                'pred': np.zeros((512, 512), dtype=np.uint8), 
                'gt': np.zeros((512, 512), dtype=np.uint8), 
                'img': np.zeros((512, 512, 3), dtype=np.uint8),
                'count': 0
            }
        gt_mask_path = os.path.join(gt_folder, file_name)
        gt_mask = np.array(Image.open(gt_mask_path))
        input_img_path = os.path.join(data_folder, 'imgs', file_name)
        input_img = np.array(Image.open(input_img_path)).astype(np.uint8) 
        pred_mask = []
        for cls_name in classes:
            if cls_name not in res_classes:
                pred_mask.append(np.zeros_like(gt_mask))
                continue
            pred_mask_path = os.path.join(result_folder, cls_name, file_name)
            pred_mask.append(np.array(Image.open(pred_mask_path)))
        # pred_mask_pos = np.stack(pred_mask, axis=0)  # (N, H, W)
        # pred_mask_neg = np.mean(1 - pred_mask_pos, axis=0, keepdims=True)  # (1, H, W)
        # pred_mask = np.concatenate([pred_mask_neg, pred_mask_pos], axis=0) / 255 # (N+1, H, W)
        # pred_mask = softmax(pred_mask)  # (N+1, H, W)
        # pred_score, pred_cls = pred_mask.max(axis=0), pred_mask.argmax(axis=0)  # (H, W), (H, W)

        pred_mask = np.stack(pred_mask, axis=0) / 255  # (N, H, W)
        pred_prob = softmax(pred_mask)  # (N, H, W)
        pred_score, pred_cls = pred_prob.max(axis=0), pred_prob.argmax(axis=0)  # (H, W), (H, W)
        pred_cls += 1  # to match gt mask which starts from 1
        pred_cls[pred_mask.max(axis=0) < 0.5] = 0  # background
        
        prev_score = copy.deepcopy(merged_res[patch_id]['score'][x:x+384, y:y+384])
        prev_cls = copy.deepcopy(merged_res[patch_id]['pred'][x:x+384, y:y+384])
        update_mask = pred_score > prev_score
        prev_cls[update_mask] = pred_cls[update_mask]
        prev_score[update_mask] = pred_score[update_mask]
        
        merged_res[patch_id]['pred'][x:x+384, y:y+384] = prev_cls
        merged_res[patch_id]['score'][x:x+384, y:y+384] = prev_score
        merged_res[patch_id]['gt'][x:x+384, y:y+384] = gt_mask
        merged_res[patch_id]['img'][x:x+384, y:y+384, :] = input_img
        merged_res[patch_id]['count'] += 1


        if merged_res[patch_id]['count'] == 4:
            final_pred = merged_res[patch_id]['pred']
            final_gt = merged_res[patch_id]['gt']
            acc = accuracy_score(final_gt, final_pred)
            overall_acc.append(acc)
            for i, cls_name in enumerate(classes):
                gt_cls_mask = (final_gt == (i + 1)).astype(np.uint8)
                pred_cls_mask = (final_pred == (i + 1)).astype(np.uint8)
                iou = iou_score(gt_cls_mask, pred_cls_mask)
                scores[cls_name].append(iou)
            
            # visualize merged predictions
            vis_lbl = np.concatenate([merged_res[patch_id]['pred'], merged_res[patch_id]['gt']], axis=1)
            vis_img = Image.fromarray(merged_res[patch_id]['img'].astype(np.uint8)).convert("RGB")

            vis_lbl = Image.fromarray(vis_lbl).convert("P")
            vis_lbl.putpalette(color_palette)
            vis_img_lbl = Image.new('RGB', (vis_img.width + vis_lbl.width, vis_img.height))
            vis_img_lbl.paste(vis_img, (0, 0))
            vis_img_lbl.paste(vis_lbl, (vis_img.width, 0))
            vis_img_lbl.save(os.path.join(vis_folder, f"{patch_id}.png"))
        
    miou = []
    res_str = ""
    for cls_name, score in scores.items():
        iou = np.mean(score)
        print(f"{cls_name}: IoU = {iou:.4f}")
        res_str += f"Class {cls_name}: IoU = {iou:.4f}\n"
        miou.append(iou)
    overall_acc = np.mean(overall_acc)
    miou = np.mean(miou)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Mean IoU: {miou:.4f}")


if __name__=="__main__":
    result_folder = "logs/syntra_rs_hameln_full/hameln/hameln"
    data_folder = "/home/yuan/data/syntra/remote_sensing/hameln"
    eval(result_folder, data_folder)

