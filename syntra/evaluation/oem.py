
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


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

def eval(result_folder, data_folder):
    """
    Evaluate segmentation results using Dice Score.
    Parameters:
        result_folder (str): Path to the folder containing N folders, each with predicted masks for one cls.
            Note that the file names should match those in ground truth folder.
        data_folder (str): syntra dataset folder containing folder 'lbls' and 'color_map.json'.
    """
    classes = sorted(os.listdir(result_folder))

    scores = {cls_name: [] for cls_name in classes}
    with open(os.path.join(data_folder, 'color_map.json'), 'r') as f:
        color_map = json.load(f)

    classes = list(color_map.keys())
    gt_folder = os.path.join(data_folder, 'lbls')
    vis_folder = os.path.join(os.path.dirname(result_folder), 'merged_vis')
    os.makedirs(vis_folder, exist_ok=True)

    test_mask_files = os.path.join(result_folder, classes[0])
    # parse patches
    full_patches = {}
    for file_name in os.listdir(test_mask_files):
        sample_name, x, y = file_name.replace('.png', '').rsplit('_', 2)
        if sample_name not in full_patches:
            full_patches[sample_name] = []
        full_patches[sample_name].append((int(x), int(y)))

    for file_name, xys in tqdm(full_patches.items()):
        xys = sorted(xys, key=lambda item: (item[0], item[1]))
        x_max, y_max = xys[-1][0] + 384, xys[-1][1] + 384
        gt_mask = np.zeros((x_max, y_max), dtype=np.uint8)
        pred_mask = np.zeros((len(classes), x_max, y_max), dtype=np.uint8)
        img = np.zeros((x_max, y_max, 3), dtype=np.uint8)

        for x, y in xys:
            gt_mask_path = os.path.join(gt_folder, file_name + f"_{x}_{y}.png")
            cur_gt_mask = np.array(Image.open(gt_mask_path)) 
            gt_mask[x:x+384, y:y+384] = cur_gt_mask

            cur_pred_masks = []
            for cls_name in classes:
                pred_mask_path = os.path.join(result_folder, cls_name, file_name + f"_{x}_{y}.png")
                cur_pred_masks.append(np.array(Image.open(pred_mask_path)))
            cur_pred_masks = np.stack(cur_pred_masks, axis=0)  # (N, H, W)
            pred_mask[:, x:x+384, y:y+384] = cur_pred_masks

            # for visualization of patches
            img_path = os.path.join(data_folder, 'imgs', file_name + f"_{x}_{y}.png")
            cur_img = np.array(Image.open(img_path).convert("RGB"))
            img[x:x+384, y:y+384] = cur_img

        pred_score, pred_mask = pred_mask.max(axis=0), pred_mask.argmax(axis=0)  # (H, W), (H, W)
        pred_mask += 1  # to match gt mask which starts from 1
        # pred_mask[pred_score < 128] = 0  # background

        for i, cls_name in enumerate(classes):
            gt_cls_mask = (gt_mask == (i + 1)).astype(np.uint8)
            pred_cls_mask = (pred_mask == (i + 1)).astype(np.uint8)
            iou = iou_score(gt_cls_mask, pred_cls_mask)
            scores[cls_name].append(iou)

        # visualize merged predictions
        gt_vis = np.zeros_like(img)
        pred_vis = np.zeros_like(img)
        for i, cls_name in enumerate(classes):
            color = np.array(color_map[cls_name])
            gt_cls_mask = (gt_mask == (i + 1))
            pred_cls_mask = (pred_mask == (i + 1))
            gt_vis[gt_cls_mask] = color
            pred_vis[pred_cls_mask] = color
        res_vis = np.concatenate([img, gt_vis, pred_vis], axis=1)
        res_vis = Image.fromarray(res_vis.astype(np.uint8))
        res_vis.save(os.path.join(vis_folder, file_name + ".png"))

    miou = []
    res_str = ""
    for cls_name, score in scores.items():
        iou = np.mean(score)
        print(f"{cls_name}: IoU = {iou:.4f}")
        res_str += f"Class {cls_name}: IoU = {iou:.4f}\n"
        miou.append(iou)
    print(f"Mean IoU over {len(classes)} classes: {np.mean(miou):.4f}")
    res_str += f"Mean IoU over {len(classes)} classes: {np.mean(miou):.4f}\n"
    with open(os.path.join(os.path.dirname(result_folder), 'eval_result.txt'), 'w') as f:
        f.write(res_str)


if __name__=="__main__":
    result_folder = "logs/syntra_remote_sensing_n4_full/remote_sensing/oem"
    data_folder = "/home/yuan/data/syntra/remote_sensing/oem"
    eval(result_folder, data_folder)

# Original without crop to 120x120:
# Class left ventricle: Mean Dice Score = 0.8206
# Class right ventricle: Mean Dice Score = 0.8059
# Class myocardium: Mean Dice Score = 0.8644

# Cropped to 120x120 and normalized to uint8: pred_size/resized
# No augmentation during training:
# Class left_ventricle: Mean Dice Score = 0.8350/0.8437
# Class right_ventricle: Mean Dice Score = 0.8234/0.8311
# Class myocardium: Mean Dice Score = 0.8793/0.8840