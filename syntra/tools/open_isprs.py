
import os
from PIL import Image
import numpy as np
from geotiff import GeoTiff


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array

color_map = {
  "forest": [34, 139, 34],          
  "grass": [124, 252, 0],     
  "settlement": [220, 20, 60],               
  "water": [30, 144, 255]
}
palette = [0, 0, 0] + sum(color_map.values(), [])

mappath = "/koko/datasets/Hameln_map_sheets_test/maps"
lblpath = "/koko/datasets/Hameln_map_sheets_test/label"
predpath = "/mars/projects20/cross_domain_syntra/logs/tmp"
outpath = "/mars/projects20/cross_domain_syntra/logs/tmp/resized"

for year in [1898, 1974, 1982, 1996]:
    filename = f"3922_{year}"
    map_img = read_tiff(os.path.join(mappath, filename + ".tif"))
    lbl_img = Image.open(os.path.join(lblpath, filename + ".png"))
    pred_img = Image.open(os.path.join(predpath,  f"hameln.{filename}.png"))

    width, height = pred_img.size
    s = 4
    w = width // s
    h = height // s

    # resize
    lbl_img = np.array(lbl_img) // 50
    lbl_img[lbl_img==5] = 4
    lbl_img = Image.fromarray(lbl_img, mode="P")
    lbl_img = lbl_img.resize((w, h))
    lbl_img.putpalette(palette)
    print(lbl_img.size)
    lbl_img.save(f"{outpath}/{filename}_gt.png")

    map_img = Image.fromarray(map_img)
    map_img = map_img.resize((w, h), Image.Resampling.LANCZOS)
    print(map_img.size)
    map_img.save(f"{outpath}/{filename}_map.png")

    pred_img = pred_img.resize((w, h))
    # pred_img.putpalette(palette)
    print(pred_img.size)
    pred_img.save(f"{outpath}/{filename}_pred.png")


