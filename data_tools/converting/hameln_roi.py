# Convert syntra dataset to version2: all hameln (bayern) data are put into one subset instead of separated folders according to map sheet.

import json
from glob import glob
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from tifffile import imread

from .syntra_data_converter import SyntraDataConverter

color_map = {
  "forest": [34, 139, 34],          
  "grass": [124, 252, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
}

german_to_english = {
    "wald": "forest",
    "grünland": "grass",
    "siedlung": "settlement",
    "fließgewässer": "water",
    "stillgewässer": "water",
}


class HamelnROIConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, crop_roi=False, redraw_grid=False,
                 fg_ratio=0.02, img_size=512, **kwargs):
        """
        Convert the GAUSS dataset of ROI regrion.

        """
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        if crop_roi:
            self.customized_data_path = self.crop_ROI(customized_data_path)
        elif redraw_grid:
            self.redraw_grid_on_vis()    
        
    def crop_ROI(self, customized_data_path):
        tif_img_dir = os.path.join(customized_data_path, 'tk_image')
        tif_lbl_dir = os.path.join(customized_data_path, 'label_tif')
        map_files = [x for x in glob(os.path.join(tif_img_dir, '*.tif')) if 'area' not in x]
        cls_name_to_id = {name: idx for idx, name in enumerate(color_map.keys(), start=1)}
        roi_data_path = os.path.join(customized_data_path, 'ROI')
        os.makedirs(os.path.join(roi_data_path, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(roi_data_path, 'lbls'), exist_ok=True)
        os.makedirs(os.path.join(roi_data_path, f'vis_win{self.syntra_img_size}'), exist_ok=True)

        xmin, xmax = 7434, 15113
        ymin, ymax = 10371, 20459
        
        for map_file in sorted(map_files):
            print(f"Processing {map_file}...")
            base_name = os.path.basename(map_file).split('.')[0]
            lbl_files = glob(os.path.join(tif_lbl_dir, f'{base_name}_*.tif'))
            label_mask_file = os.path.join(tif_lbl_dir, f'{base_name}.tif')
            
            lbl_mask = imread(label_mask_file)
            # xs, ys = np.where(lbl_mask == 1)  # foreground pixels
            # xmin, xmax = xs.min(), xs.max()
            # ymin, ymax = ys.min(), ys.max()

            img = imread(map_file)
            img = img[xmin:xmax+1, ymin:ymax+1]
            lbl_merged = np.zeros_like(lbl_mask[xmin:xmax+1, ymin:ymax+1], dtype=np.uint8)
            for lf in lbl_files:
                cls_name = os.path.basename(lf).split('.')[0].replace(f'{base_name}_', '')
                cls_name = german_to_english[cls_name]
                lbl = imread(lf)
                lbl = lbl[xmin:xmax+1, ymin:ymax+1]
                lbl_merged[lbl == 1] = cls_name_to_id[cls_name]
            
            # save cropped images and labels to a new folder
            img_pil = Image.fromarray(img)
            lbl_pil = self._convert_annotation(lbl_merged)
            img_pil.save(os.path.join(roi_data_path, 'imgs', f'{base_name}.png'))
            lbl_pil.save(os.path.join(roi_data_path, 'lbls', f'{base_name}.png'))
            # save visualization of image and label side by side
            vis_width = img_pil.width // 4
            vis_height = img_pil.height // 4
            vis_img = img_pil.resize((vis_width, vis_height))
            # draw tiling lines and indices on the visualization image
            vis_img = draw_tiling_grid(
                vis_img,
                tile_size=(self.syntra_img_size // 4, self.syntra_img_size // 4),
                stride=(self.syntra_img_size // 4, self.syntra_img_size // 4),
                line_color=(255, 0, 0),
                line_width=5,
                text_color=(255, 255, 255),
                text_bg=(0, 0, 0),
                text_padding=3,
                font_size=32,
                index_mode="rc",
            )
            vis = Image.new('RGB', (vis_width * 2, vis_height))
            vis.paste(vis_img, (0, 0))
            vis.paste(lbl_pil.resize((vis_width, vis_height)).convert('RGB'), (vis_width, 0))
            vis.save(os.path.join(roi_data_path, f'vis_win{self.syntra_img_size}'), f'{base_name}.png')
            
        return roi_data_path
    
    def redraw_grid_on_vis(self):
        img_dir = os.path.join(self.customized_data_path, 'imgs')
        vis_dir = os.path.join(self.customized_data_path, f'vis_win{self.syntra_img_size}')
        os.makedirs(vis_dir, exist_ok=True)
        for img_file in sorted(glob(os.path.join(img_dir, '*.png'))):
            img = Image.open(img_file)
            vis_img = img.resize((img.width // 4, img.height // 4))
            vis_img = draw_tiling_grid(
                vis_img,
                tile_size=(self.syntra_img_size // 4, self.syntra_img_size // 4),
                stride=(self.syntra_img_size // 4, self.syntra_img_size // 4),
                line_color=(255, 0, 0),
                line_width=5,
                text_color=(255, 255, 255),
                text_bg=(0, 0, 0),
                text_padding=3,
                font_size=32,
                index_mode="rc",
            )   
            vis_lbl = Image.open(
                os.path.join(self.customized_data_path, 'lbls', os.path.basename(img_file))
                ).resize((img.width // 4, img.height // 4)).convert('RGB')
            vis = Image.new('RGB', (vis_img.width * 2, vis_img.height))
            vis.paste(vis_img, (0, 0))
            vis.paste(vis_lbl, (vis_img.width, 0))
            vis.save(os.path.join(vis_dir, os.path.basename(img_file)))


    def _parse_split_img_lbl(self, split):
        assert split in ['test']
        samples = []
        split_samples = os.listdir(os.path.join(self.customized_data_path, 'imgs'))
        for sample in split_samples:
            img_file = os.path.join(self.customized_data_path, 'imgs', sample)
            lbl_file = os.path.join(self.customized_data_path, 'lbls', sample)
            samples.append((img_file, lbl_file))
        return samples

    def select_samples(self, n):
        samples_dict = {}
        for subset in self.syntra1_subsets:
            filename = os.path.join(self.customized_data_path, subset, f'cls_selection_{n}.json')
            with open(filename, 'r') as f:
                cls_samples = json.load(f)
            for cls, samples in cls_samples.items():
                if cls not in samples:
                    samples[cls] = []
                samples_dict[cls].extend(samples)

        output_filename = os.path.join(self.syntra_data_path, subset, f's{n}.json')
        with open(output_filename, 'w') as f:
            json.dump(samples_dict, f, indent=4)
    
    def _load_original_lbl(self, lbl_file):
        """Load tif image file."""
        return Image.open(lbl_file)
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).split('.')[0] 
        return base_name  
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(self.color_map) + 1)}
    

    

    

def draw_tiling_grid(
    img: Image.Image,
    tile_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
    line_color=(255, 0, 0),
    line_width: int = 2,
    text_color=(255, 255, 255),
    text_bg=(0, 0, 0),
    text_padding: int = 3,
    font: ImageFont.ImageFont | None = None,
    font_size: int = 32,  # <- add this
    index_mode: str = "rc",  # "rc" or "linear"
) -> Image.Image:
    """
    Draw a tiling grid and per-tile indices over a PIL image.

    tile_size: (tile_w, tile_h)
    stride: (sx, sy). If None, defaults to tile_size (no overlap).
    origin: top-left offset where tiling starts.
    index_mode:
      - "rc": draws "r,c"
      - "linear": draws a single increasing integer id
    """
    if img.mode != "RGB":
        vis = img.convert("RGB").copy()
    else:
        vis = img.copy()

    draw = ImageDraw.Draw(vis)

    tile_w, tile_h = tile_size
    sx, sy = stride if stride is not None else tile_size
    ox, oy = origin
    W, H = vis.size

    if tile_w <= 0 or tile_h <= 0 or sx <= 0 or sy <= 0:
        raise ValueError("tile_size and stride must be positive")

    if font is None:
        # Try a reasonable default; fall back to PIL default if unavailable.
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()

    # Tile iteration bounds (covering tiles that start within the image extent)
    xs = list(range(ox, W, sx))
    ys = list(range(oy, H, sy))

    # Draw grid lines (tile boundaries)
    for x in xs:
        # Boundary at x and x+tile_w
        draw.line([(x, 0), (x, H)], fill=line_color, width=line_width)
        xb = x + tile_w
        if 0 <= xb <= W:
            draw.line([(xb, 0), (xb, H)], fill=line_color, width=line_width)

    for y in ys:
        draw.line([(0, y), (W, y)], fill=line_color, width=line_width)
        yb = y + tile_h
        if 0 <= yb <= H:
            draw.line([(0, yb), (W, yb)], fill=line_color, width=line_width)

    # Draw tile indices
    tile_id = 0
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            # Only label tiles that overlap the image (common for edge tiles)
            if x >= W or y >= H:
                continue

            label = f"{r},{c}" if index_mode == "rc" else str(tile_id)
            tile_id += 1

            # Place label near tile top-left, clamped into image
            tx = max(0, min(W - 1, x + text_padding))
            ty = max(0, min(H - 1, y + text_padding))

            # Text background box for readability
            bbox = draw.textbbox((tx, ty), label, font=font)
            bx0, by0, bx1, by1 = bbox
            bx0 -= text_padding
            by0 -= text_padding
            bx1 += text_padding
            by1 += text_padding

            # Clamp background rect
            bx0 = max(0, bx0); by0 = max(0, by0)
            bx1 = min(W, bx1); by1 = min(H, by1)

            draw.rectangle([bx0, by0, bx1, by1], fill=text_bg)
            draw.text((tx, ty), label, fill=text_color, font=font)

    return vis



    



