import os
import numpy as np
from tifffile import imread
from PIL import Image
from glob import glob


german_to_english = {
    "wald": "forest",
    "grünland": "grass",
    "siedlung": "settlement",
    "fließgewässer": "water",
    "stillgewässer": "water",
}

color_map = {
  "forest": [34, 139, 34],          
  "grass": [124, 252, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
}

cls_name_to_id = {name: idx+1 for idx, name in enumerate(color_map.keys())}
palette = [0, 0, 0] + sum(color_map.values(), [])  # adding background color

def map_crop_and_save_ROI(map_file, save_path):
    tif_lbl_dir = os.path.dirname(map_file).replace('tk_image', 'label_tif')
    base_name = os.path.basename(map_file).split('.')[0]
    label_mask_file = os.path.join(tif_lbl_dir, f'{base_name}.tif')
    
    lbl_mask = imread(label_mask_file)
    xs, ys = np.where(lbl_mask == 1)  # foreground pixels
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    img = imread(map_file)
    img = img[xmin:xmax+1, ymin:ymax+1, :]

    img = Image.fromarray(img).save(os.path.join(save_path, f'{base_name}.tif'))    

    lbl_files = glob(os.path.join(tif_lbl_dir, f'{base_name}_*.tif'))
    lbl_merged = np.zeros_like(lbl_mask[xmin:xmax+1, ymin:ymax+1], dtype=np.uint8)
    for lf in lbl_files:
        cls_name = os.path.basename(lf).split('.')[0].replace(f'{base_name}_', '')
        cls_name = german_to_english[cls_name]
        lbl = imread(lf)
        lbl = lbl[xmin:xmax+1, ymin:ymax+1]
        lbl_merged[lbl == 1] = cls_name_to_id[cls_name]
    lbl_merged_img = Image.fromarray(lbl_merged)
    lbl_merged_img.putpalette(palette)
    lbl_merged_img.save(os.path.join(save_path, f'{base_name}_lbl.png'))


def prob_to_pred(img_path, save_path):
    if img_path.endswith('.tif'):
        img = imread(img_path)
        pred_mask = np.array(img) > 0  # thresholding to get binary mask
    else:
        img = Image.open(img_path)
        pred_mask = np.array(img) > 128  # thresholding to get binary mask
    
    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
    pred_mask_img.save(save_path)

if __name__ == "__main__":

    # map_crop_and_save_ROI('/home/yuan/data/HisMap/ikg_ml_dataset/tk_image/1960.tif', '/home/yuan/data/syntra/test_hameln_ROI/data')
    # prob_to_pred(
    #     '/koko/SYNTRA_logs/smol/smol_siegfried_railway_n1_s10/railway/railway/railway/111_sie.png', 
    #     '/home/yuan/Downloads/111_sie_smol_pred.png')

    # convert a probability map to a binary mask with a given color
    color_map = {
  "forest": [34, 139, 34],          
  "grass": [221, 223, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
    }
    img = Image.open('logs/smol_hismap_n1_full/hismap,siegfried/bayern.489_1910/forest/384_1920.png')
    alpha = np.array(img)
    color = color_map['forest']
    color_mask = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    color_mask[alpha > 128] = color  # apply color where alpha is above threshold
    color_mask_img = Image.fromarray(color_mask)
    color_mask_img.save('logs/smol_hismap_n1_full/hismap,siegfried/bayern.489_1910/forest/384_1920_color.png')