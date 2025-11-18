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


def syncronize_two_folders(folder_src, folder_tgt):
    """Sync all files from folder_src to folder_tgt recursively."""
    import subprocess
    subprocess.run(f"rsync -av --progress {folder_src}/ {folder_tgt}/", shell=True, check=True)


def syncronize_to_remote(folder_src, folder_tgt, remote_user, remote_host, ignore_list=None):
    """Sync all files from folder_src to folder_tgt on a remote machine recursively."""
    import subprocess
    if ignore_list:
        ignore_args = ' '.join([f"--exclude='{item}'" for item in ignore_list])
        subprocess.run(f"rsync -av --progress {ignore_args} {folder_src}/ {remote_user}@{remote_host}:{folder_tgt}/", shell=True, check=True)
    else:
        subprocess.run(f"rsync -av --progress {folder_src}/ {remote_user}@{remote_host}:{folder_tgt}/", shell=True, check=True)


def syncronize_from_remote(folder_src, folder_tgt, remote_user, remote_host, ignore_list=None):
    """Sync all files from folder_src on a remote machine to folder_tgt recursively."""
    
    if ignore_list:
        ignore_args = ' '.join([f"--exclude='{item}'" for item in ignore_list])
        subprocess.run(f"rsync -av --progress {ignore_args} {remote_user}@{remote_host}:{folder_src}/ {folder_tgt}/", shell=True, check=True)
    else:
        subprocess.run(f"rsync -av --progress {remote_user}@{remote_host}:{folder_src}/ {folder_tgt}/", shell=True, check=True)


def sync_to_kisski(local_path, remote_path, uid, remote_host_shortcut, is_code_repo=False):
    assert remote_host_shortcut in ["Emmy", "Emmy-p2", "Emmy-p3", "KISSKI"], "Unknown remote host."
    assert uid.startswith("u"), "User ID should start with 'u'"
    if is_code_repo:
        ignore_list = [".git/", "__pycache__/", "*.pyc", ".vscode/", ".idea/", ".DS_Store"]
        # read from gitignore
        if os.path.exists(os.path.join(local_path, ".gitignore")):
            with open(os.path.join(local_path, ".gitignore"), "r") as f:
                ignore_list += [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
        exclude_args = ' '.join([f"--exclude='{item}'" for item in ignore_list])
        cmd = [
            "rsync", "-av", "--progress", "-e", "ssh",
            exclude_args,
            f"{local_path}/",
            f"{remote_host_shortcut}:{remote_path}/"
        ]
        subprocess.run(' '.join(cmd), shell=True, check=True)
    else:
        subprocess.run(f"rsync -av --progress -e ssh {local_path}/ {uid}@{remote_host_shortcut}:{remote_path}/", shell=True, check=True)


def sync_from_kisski(local_path, remote_path, uid, remote_host_shortcut):
    assert remote_host_shortcut in ["Emmy", "Emmy-p2", "Emmy-p3", "KISSKI"], "Unknown remote host."
    assert uid.startswith("u"), "User ID should start with 'u'"
    subprocess.run(f"rsync -av --progress -e ssh {uid}@{remote_host_shortcut}:{remote_path}/ {local_path}/", shell=True, check=True)



if __name__ == "__main__":

    sync_to_kisski(
        local_path="/home/yuan/data/syntra/forest_monitoring",
        remote_path="/mnt/vast-kisski/projects/kisski-gauss-zentrum/syntra_data/forest_monitoring",
        uid="u23994",
        remote_host_shortcut="KISSKI",
        is_code_repo=False
    )

    # sync_from_kisski(
    #     local_path="logs/syntra_medical_n4_full",
    #     remote_path="/mnt/vast-kisski/projects/kisski-gauss-zentrum/logs/syntra_medical_n4_full",
    #     uid="u23994",
    #     remote_host_shortcut="KISSKI"
    # )

    # syncronize_two_folders(
    #     "/home/yuan/data/syntra/hismap",
    #     "/koko/datasets/syntra/hismap")

    # local_path = "/home/yuan/code/CD_SYNTRA"
    # remote_path = "/home/yuang/data/SYNTRA"
    # # read from gitignore
    # with open(os.path.join(local_path, ".gitignore"), "r") as f:
    #     ignore_list = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

    # syncronize_to_remote(
    #     "/home/yuan/data/syntra/autonomous_driving/cityscapes",
    #     "/home/yuang/data/syntra/autonomous_driving/cityscapes",
    #     "yuang",
    #     "130.75.51.199",
    #     ignore_list=["imgs/", "lbls/"]
    # )

    # syncronize_from_remote(
    #     "/home/yuan/data/syntra/hismap",
    #      "/koko/datasets/syntra/hismap",
    #     "yuan",
    #     "130.75.51.193"
    # )

    # list_files("/home/yuan/data/OpenEarthMap/xview2")

    # syncronize_to_remote(
    #     "/home/yuan/data/syntra/autonomous_driving",
    #     "data/syntra/autonomous_driving",
    #     "yuang",
    #     "130.75.51.199"
    # )