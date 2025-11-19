import numpy as np
from PIL import Image
from tifffile import imread
from pathlib import Path
import os
import subprocess


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def float32_to_rgba_bytes(arr: np.ndarray) -> np.ndarray:
    """
    Pack a 2D float32 array into a 3D uint8 RGBA array (H, W, 4), little-endian.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape {arr.shape}")
    # Ensure little-endian float32
    arr_le = np.asarray(arr, dtype=np.dtype("<f4"), order="C")
    # View bytes and reshape to RGBA
    rgba = np.ascontiguousarray(arr_le).view(np.uint8).reshape(arr_le.shape + (4,))
    return rgba


def tiff32_to_png_rgba(tiff_path: str | Path, png_path: str | Path) -> None:
    """
    Read float32 grayscale TIFF and write RGBA PNG with exact byte packing.
    """
    tiff_path = Path(tiff_path)
    png_path = Path(png_path)

    arr = imread(tiff_path)

    if arr.ndim != 2:
        raise ValueError(f"TIFF must be 2D grayscale, got shape {arr.shape}")

    rgba = float32_to_rgba_bytes(arr)
    im_rgba = Image.fromarray(rgba, mode="RGBA")
    # Save PNG; compression/filtering won't alter pixel bytes
    im_rgba.save(png_path, optimize=False, compress_level=9)


def tif_to_png():
    in_path = "/home/yuan/data/syntra/geology/seismic_facies.f3/imgs_tif"
    out_path = "/home/yuan/data/syntra/geology/seismic_facies.f3/imgs"

    for tiff_file in Path(in_path).glob("*.tiff"):
        png_file = Path(out_path) / (tiff_file.stem + ".tiff")
        tif_img = imread(tiff_file)
        pil_img = Image.fromarray(tif_img)
        pil_img.save(png_file, compression="tiff_lzw")

        png_img = np.array(Image.open(png_file), dtype=np.float32)
        arr2 = np.array(Image.open(tiff_file), dtype=np.float32)

        print((tif_img == png_img).all())
        print((arr2 == tif_img).all())


def resize_imgs_and_lbls(dataset_path, size=384):
    img_files = list(Path(dataset_path).glob("imgs/*"))
    lbl_files = list(Path(dataset_path).glob("lbls/*"))
    for img_file, lbl_file in zip(img_files, lbl_files):
        img = Image.open(img_file)
        lbl = Image.open(lbl_file)

        img_resized = img.resize((size, size), resample=Image.BILINEAR)
        lbl_resized = lbl.resize((size, size), resample=Image.NEAREST)

        img_resized.save(img_file)
        lbl_resized.save(lbl_file)


if __name__ == "__main__":

    resize_imgs_and_lbls("/home/yuan/data/syntra/medical/medical.acdc", size=384)