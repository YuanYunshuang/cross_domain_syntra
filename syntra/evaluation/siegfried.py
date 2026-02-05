import numpy as np
from PIL import Image
import os


pred_path = "logs/smol_hismap_n1_full/hismap,siegfried/railway/railway"
gt_path = "/home/yuan/data/syntra/siegfried/railway/lbls"

files = os.listdir(pred_path)

n = 0
tp = 0
fp = 0
tn = 0
overall_tp = 0
union = 0
pos_pred = 0
pos_gt = 0

for f in files:
    pred = Image.open(os.path.join(pred_path, f))
    gt = Image.open(os.path.join(gt_path, f))

    # resize pred to gt size with interpolation
    pred = pred.resize(gt.size, resample=Image.Resampling.BILINEAR)
    pred = np.array(pred) >= 128
    gt = np.array(gt) == 1
    n += 1

    # calculate iou
    tp += np.sum(pred & gt)
    overall_tp += (pred == gt).sum() / gt.size
    union += np.sum(np.logical_or(pred, gt))
    pos_pred += np.sum(pred)
    pos_gt += np.sum(gt)


iou = tp / union 
prec = tp / pos_pred 
rec = tp / pos_gt 
acc = overall_tp / n

print(
    f"IoU: {iou*100:.1f},\n Precision: {prec*100:.1f},\n Recall: {rec*100:.1f},\n Accuracy: {acc*100:.1f}"
)