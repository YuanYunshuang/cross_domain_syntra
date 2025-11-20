
import os
import json
import numpy as np
from PIL import Image


def dice_score(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Score between the ground truth and predicted masks.

    Parameters:
    y_true (ndarray): Ground truth binary mask.
    y_pred (ndarray): Predicted binary mask.
    smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    float: Dice Score.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def eval(result_folder, data_folder):
    """
    Evaluate segmentation results using Dice Score.
    Parameters:
        result_folder (str): Path to the folder containing N folders, each with predicted masks for one cls.
            Note that the file names should match those in ground truth folder.
        data_folder (str): syntra dataset folder containing folder 'lbls' and 'color_map.json'.
    """
    classes = sorted(os.listdir(result_folder))

    dice_scores = {cls_name: [] for cls_name in classes}
    with open(os.path.join(data_folder, 'color_map.json'), 'r') as f:
        color_map = json.load(f)

    classes = list(color_map.keys())
    gt_folder = os.path.join(data_folder, 'lbls')
    for cls_name in classes:
        assert cls_name in classes, f"Class {cls_name} not found in result folder."

        cls_idx = classes.index(cls_name)
        pred_cls_folder = os.path.join(result_folder, cls_name)
        
        for file_name in os.listdir(pred_cls_folder):
            pred_mask_path = os.path.join(pred_cls_folder, file_name)
            gt_mask_path = os.path.join(gt_folder, file_name)

            if not os.path.exists(gt_mask_path):
                print(f"Ground truth mask for {file_name} not found, skipping.")
                continue

            pred_mask = Image.open(pred_mask_path) 
            gt_mask = Image.open(gt_mask_path) 
            if pred_mask.size != gt_mask.size:
                pred_mask = pred_mask.resize(gt_mask.size, resample=Image.BILINEAR)

            pred_mask = np.array(pred_mask) > 128
            gt_mask = np.array(gt_mask) == cls_idx + 1

            dice = dice_score(gt_mask, pred_mask)
            dice_scores[cls_name].append(dice)
        
        print(f"Class {cls_name}: Mean Dice Score = {np.mean(dice_scores[cls_name]):.4f}")


if __name__=="__main__":
    result_folder = "logs/syntra_acdc_n3_full/acdc/medical.acdc"
    data_folder = "/home/yuan/data/syntra_original_size/medical/medical.acdc"
    eval(result_folder, data_folder)

# Class left ventricle: Mean Dice Score = 0.8206
# Class right ventricle: Mean Dice Score = 0.8059
# Class myocardium: Mean Dice Score = 0.8644