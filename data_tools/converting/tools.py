import numpy as np
from PIL import Image
from tifffile import imread
from pathlib import Path
import os


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


def syncronize_two_folders(folder_src, folder_tgt):
    """Sync all files from folder_src to folder_tgt recursively."""
    import subprocess
    subprocess.run(f"rsync -av --progress {folder_src}/ {folder_tgt}/", shell=True, check=True)


if __name__ == "__main__":


    syncronize_two_folders(
        "/home/yuan/data/syntra/forest_monitoring",
        "/koko/datasets/syntra/forest_monitoring")
