import os
import subprocess


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

def copy_to_remote(file_src, file_tgt, remote_user, remote_host):
    """Copy a file from file_src to file_tgt on a remote machine."""
    import subprocess
    subprocess.run(f"scp {file_src} {remote_user}@{remote_host}:{file_tgt}", shell=True, check=True)


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

    # sync_to_kisski(
    #     local_path="/home/yuan/data/syntra/medical/acdc",
    #     remote_path="/mnt/vast-kisski/projects/kisski-gauss-zentrum/syntra_data/medical/acdc",
    #     uid="u23994",
    #     remote_host_shortcut="KISSKI",
    #     is_code_repo=False
    # )

    # sync_from_kisski(
    #     local_path="/koko/SYNTRA_logs",
    #     remote_path="/mnt/vast-kisski/projects/kisski-gauss-zentrum/logs",
    #     uid="u23994",
    #     remote_host_shortcut="KISSKI",
    # )

    # sync_from_kisski(
    #     local_path="/koko/SYNTRA_logs",
    #     remote_path="/mnt/vast-kisski/projects/kisski-gauss-zentrum/logs",
    #     uid="u23994",
    #     remote_host_shortcut="KISSKI"
    # )

    syncronize_two_folders(
        "/home/yuan/data/syntra/hismap",
        "/koko/datasets/syntra/hismap")

    # local_path = "/home/yuan/code/CD_SYNTRA"
    # remote_path = "/home/yuang/data/SYNTRA"
    # # read from gitignore
    # with open(os.path.join(local_path, ".gitignore"), "r") as f:
    #     ignore_list = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

    # copy_to_remote(
    #     "/koko/datasets/syntra/forest_monitoring/trees_september/pairing_train_samples_train_samples_intracls.json",
    #     "/home/yuang/data/syntra/",
    #     "yuang",
    #     "130.75.51.199",
    #     # ignore_list=["imgs/", "lbls/"]
    # )

    # syncronize_from_remote(
    #     "/home/yuan/data/syntra_original_size/medical/acdc",
    #      "/koko/datasets/syntra/medical/acdc",
    #     "yuan",
    #     "130.75.51.193"
    # )

    # list_files("/home/yuan/data/OpenEarthMap/xview2")

    # syncronize_to_remote(
    #     "/home/yuan/data/syntra/hismap",
    #     "data/syntra/hismap",
    #     "yuang",
    #     "130.75.51.199"
    # )
