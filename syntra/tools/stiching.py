# Stiching the results generated

import os

import numpy as np
from PIL import Image

color_map = {
    "forest": [34, 139, 34],
    "grass": [124, 252, 0],
    "settlement": [220, 20, 60],
    "water": [30, 144, 255],
}

# Index 0 is reserved for background in the output mask.
color_palette = [0, 0, 0] + sum(color_map.values(), [])
color_palette += [0] * (768 - len(color_palette))


def stiching_results(results_dir, output_dir, img_size=384):
    """
    Stiching the results generated.

    :param results_dir: directory where one map sheet result is stored
    :param output_dir: directory where the stitched result will be stored
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = list(color_map.keys())
    img_root = os.path.join(results_dir, classes[0])
    img_files = sorted(
        img for img in os.listdir(img_root) if os.path.isfile(os.path.join(img_root, img)) 
    )
    if not img_files:
        raise ValueError(f"No tile images found in {img_root}.")

    tile_positions = [
        tuple(map(int, os.path.splitext(img)[0].split("_")[:2])) for img in img_files
    ]
    max_x = max(x for x, _ in tile_positions) + img_size
    max_y = max(y for _, y in tile_positions) + img_size
    out_img = Image.new("P", (max_x, max_y), color=0)
    out_img.putpalette(color_palette)

    for img, (y, x) in zip(img_files, tile_positions):
        preds = []
        for cls in classes:
            img_path = os.path.join(results_dir, cls, img)
            if os.path.isfile(img_path):
                with Image.open(img_path) as img_pil:
                    preds.append(np.array(img_pil))
            else:
                preds.append(np.zeros((img_size, img_size), dtype=np.uint8))
        
        preds = np.stack(preds, axis=0)
        pred_cls = np.argmax(preds, axis=0) + 1
        pred_cls_max = np.max(preds, axis=0)
        pred_cls[pred_cls_max < 128] = 0

        pred_img = Image.fromarray(pred_cls.astype(np.uint8), mode="P")
        pred_img.putpalette(color_palette)
        out_img.paste(pred_img, (x, y))

    out_img.save(os.path.join(output_dir, f"{os.path.basename(results_dir)}.png"))


if __name__ == "__main__":
    results_dir = '/koko/SYNTRA_logs/smol/smol_hameln_n1_full/hameln_test/hameln.3821_1974'
    output_dir = '/koko/SYNTRA_logs/smol/smol_hameln_n1_full/stiched_results'
    stiching_results(results_dir, output_dir)
