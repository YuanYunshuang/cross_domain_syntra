import os
import shutil

import numpy as np
import matplotlib.pyplot as plt


def plot_loss(*res_dirs):
    for res_dir in res_dirs:
        if 'from-scratch' in res_dir or 'train-on-all-data' in res_dir:
            continue
        dirname = os.path.basename(res_dir)
        filename = os.path.join(res_dir, f"medsam_loss.txt")
        shutil.copy(filename, filename.replace("medsam_", ""))
        filename = os.path.join(res_dir, f"loss.txt")

        losses = np.loadtxt(filename, delimiter=',')
        xticks = np.arange(len(losses))

        plt.plot(xticks, losses[:, 1], label=dirname)

    plt.legend()
    plt.savefig("/home/yuan/Downloads/tmp.jpg")
    plt.close()


if __name__ == "__main__":
    logdir = "/home/yuan/code/HisMap/smolsam/logs"
    plot_loss(*[os.path.join(logdir, d) for d in os.listdir(logdir)])
    # plot_loss(
    #     os.path.join(logdir, "1e-5.use-amp-20250418-0345"),
    #     os.path.join(logdir, "cosine_lr1e-5-20250416-1039"),
    #     os.path.join(logdir, "5e-5.train-patch-emb"),
    # )