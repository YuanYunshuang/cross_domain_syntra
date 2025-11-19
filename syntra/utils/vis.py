import os

from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from functools import partial


def rgb_img_to_gray_med_img(rgb_array, normalize=True, to_uint8=True):
    """
    Convert RGB image to medical grayscale image.
    Parameters
    ----------
    rgb_array: np.array

    Returns
    -------
    img: np.array
        Grayscale image array. Shape (H, W) with min 0 and max 2**24-1
    """
    r = rgb_array[..., 0].astype(np.uint32)
    g = rgb_array[..., 1].astype(np.uint32)
    b = rgb_array[..., 2].astype(np.uint32)
    img = r + (g << 8) + (b << 16)
    img = img.astype(np.float32)
    if normalize:
        img = (img - img.min()) / (img.max() - img.min()) 
    if to_uint8:
        img = (img * 255).astype(np.uint8)
    # one channel to three channels for vis
    img = np.stack([img, img, img], axis=-1)
    return img


def overlay_masks_on_image(image, tgt_name, alpha=0.5, colors=None):    
    """
    Overlay segmentation masks on an image.

    Parameters:
    - image (np.ndarray): The original image (H, W, 3), values in [0, 255]
    - alpha (np.ndarray): Transparency factor for the masks
    Returns:
    - np.ndarray: Image with masks overlaid (H, W, 3)
    """
    overlaid_image = image.copy()
    num_masks = alpha.shape[0]

    # Determine color map based on tgt_name
    if colors is None:
        # Default to random colors if no specific mapping found
        colors = [(np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)) for _ in range(num_masks)]

    if len(colors) == num_masks - 1:
        # add background color
        colors = [(0, 0, 0)] + colors
    for i in range(num_masks):
        color = colors[i % len(colors)]
        mask = np.ones_like(alpha[i]).astype(bool)
        # mask = np.zeros_like(overlaid_image[..., 0]).astype(bool) 
        overlaid_image = apply_colored_mask(overlaid_image, mask, color, alpha[i])

    return overlaid_image


def save_results(out_dir, img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, meta, rescale=0.5, color_map=None):
    # overlay_func = partial(overlay_masks_on_image, colors=colors)
    B, T, _, H, W = img_src.shape
    N = lbl_src.shape[2]

    os.makedirs(out_dir, exist_ok=True)
    lbl_src = F.interpolate(
        lbl_src.flatten(0, 1).float(),
        img_src.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).view(B, T, N, H, W)

    img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy() * 0.25 + 0.5
    img_src = (img_src * 255).astype(np.uint8)
    img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy() * 0.25 + 0.5
    img_tgt = (img_tgt * 255).astype(np.uint8)
    lbl_src = lbl_src.detach().cpu().numpy()
    lbl_tgt = lbl_tgt.detach().cpu().numpy()
    pred_tgt = pred_tgt.detach().sigmoid().cpu().numpy()
    names = [m['tgt_name'] for m in meta]

    for i in range(1):
        cur_classes = [x.split('/')[1] for x in meta[i]['cls_name']]
        cur_colors = [color_map[cls] for cls in cur_classes]
        midbar = np.zeros_like(img_tgt[0, :, :10]) + 128
        row1, row2 = [], []
        for t in range(T):
            if 'chaos' or 'acdc' in names[i]:
                # convert to gray medical image
                img_src[i, t] = rgb_img_to_gray_med_img(img_src[i, t])
            row1.append(img_src[i, t])
            row1.append(midbar)
            overlayed_img = overlay_masks_on_image(img_src[i, t], names[i], 
                                                   alpha=lbl_src[i, t] * 0.5, 
                                                   colors=cur_colors)
            row2.append(overlayed_img)
            row2.append(midbar)
        if 'chaos' in names[i]:
            img_tgt[i] = rgb_img_to_gray_med_img(img_tgt[i])
        row1.append(img_tgt[i])
        row1.append(midbar)
        overlayed_tgt_img = overlay_masks_on_image(img_tgt[i], names[i], 
                                                   alpha=lbl_tgt[i] * 0.5,
                                                   colors=cur_colors)
        row2.append(overlayed_tgt_img)
        row2.append(midbar)

        pred_prob = pred_tgt[i].squeeze(1) # (N, H, W)
        prob_bg = 1 - pred_prob.max(axis=0)[None, ...]  # (1, H, W)
        pred_prob_norm = np.concatenate([prob_bg, pred_prob], axis=0)  # (N+1, H, W)
        pred_prob_norm = pred_prob_norm / (pred_prob_norm.sum(axis=0, keepdims=True) + 1e-8)
        pred_inds = np.argmax(pred_prob_norm, axis=0)  # (H, W)
        pred = np.zeros_like(pred_prob, dtype=bool)  # (N, H, W)
        for n in range(N):
            pred[n] = (pred_inds == n + 1)

        overlayed_pred_prob_img = overlay_masks_on_image(img_tgt[i], names[i], 
                                                         alpha=pred_prob_norm,
                                                         colors=cur_colors)
        row1.append(overlayed_pred_prob_img)
        overlayed_pred_img = overlay_masks_on_image(img_tgt[i],  names[i], 
                                                    alpha=pred * 0.5,
                                                    colors=cur_colors)
        row2.append(overlayed_pred_img)

        row1 = np.concatenate(row1, axis=1)[..., ::-1]
        row2 = np.concatenate(row2, axis=1)[..., ::-1]
        midbar = np.zeros_like(row1[:10, :]) + 128
        img = np.concatenate([row1, midbar, row2], axis=0)
        try:
            if rescale != 1.0:
                img = Image.fromarray(img)
                img = img.resize((int(img.width * rescale), int(img.height * rescale)), 
                                 resample=Image.Resampling.BILINEAR)
                img.save(os.path.join(out_dir, names[i] + f"_{','.join(cur_classes)}.png"))
        except:
            print(f"Failed to save {names[i]}")


def save_results_v1(out_dir, img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, names, label_only=False):
    bs = pred_tgt.shape[0]

    if label_only:
        inf_dir = os.path.join(out_dir, f'inf_labels')
        os.makedirs(inf_dir, exist_ok=True)
        pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()
        pred_tgt = (pred_tgt * 255).astype(np.uint8)
        for i in range(bs):
            img = Image.fromarray(pred_tgt[i])
            img.save(os.path.join(inf_dir, names[i] + '.png'))
    else:
        inf_dir = os.path.join(out_dir, f'inf_plots')
        os.makedirs(inf_dir, exist_ok=True)
        lbl_src = F.interpolate(
            lbl_src.float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        lbl_src = lbl_src.view(img_src.shape[0], img_src.shape[1], img_src.shape[3], img_src.shape[4])
        T = img_src.shape[1] + 1

        img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy() * 0.5 + 0.5
        img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        lbl_src = lbl_src.detach().cpu().numpy()
        lbl_tgt = lbl_tgt.squeeze(1).detach().cpu().numpy()
        pred_tgt = pred_tgt.detach().sigmoid().squeeze(1).squeeze(1).cpu().numpy()

        for i in range(1):
            midbar = np.zeros_like(img_tgt[0, :, :10]) + 0.5
            imgs = []
            for t in range(T-1):
                imgs.append(img_src[i, t])
                imgs.append(midbar)
                imgs.append(lbl_src[i, t][..., None][..., [0, 0, 0]])
                imgs.append(midbar)
            
            imgs.extend([
                img_tgt[i], midbar,
                pred_tgt[i][..., None][..., [0, 0, 0]], midbar,
                lbl_tgt[i][..., None][..., [0, 0, 0]],
            ])
            try:
                img = np.concatenate(imgs, axis=1)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(inf_dir, names[i] + '.png'))
            except:
                print(f"Failed to save {names[i]}")


def vis_train(img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, aux, iter_num, size_per_img=1.0):
    assert img_src.dim() == 5  # (bs, T, 3, H, W)
    lbl_src = F.interpolate(
            lbl_src.flatten(0, 1).float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    bs, nsrc, _, H, W = img_src.shape
    n_notions = lbl_tgt.shape[1]
    lbl_src = lbl_src.view(bs, nsrc, n_notions, H, W)
    T = img_src.shape[1] + 1

    ncol = T + 1
    vis_img = np.zeros((bs * H // 2, ncol * W // 2, 3), dtype=np.uint8)
    # generate random colors for each notion
    if 'cls_color' in aux[0]:
        colors = [x['cls_color'] for x in aux]
    else:
        colors = [np.random.randint(0, 255, size=(n_notions, 3), dtype=np.uint8)] * bs

    # half the size for visualization
    img_src = img_src[..., ::2, ::2]
    img_tgt = img_tgt[..., ::2, ::2]
    lbl_src = lbl_src[..., ::2, ::2]
    lbl_tgt = lbl_tgt[..., ::2, ::2]
    pred_tgt = pred_tgt[..., ::2, ::2]

    # unnormalize images
    img_src = img_src * 0.25 + 0.5
    img_tgt = img_tgt * 0.25 + 0.5
    img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy() # (bs, T, H, W, 3)
    img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy() # (bs, H, W, 3)
    lbl_src = lbl_src.detach().cpu().numpy() # (bs, T, n_notions, H, W)
    lbl_tgt = lbl_tgt.squeeze(1).detach().cpu().numpy() # (bs, n_notions, H, W)
    pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()

    for i in range(bs):
        for t in range(T-1):
            vis_img[i*H//2:(i+1)*H//2, t*W//2:(t+1)*W//2] = overlay_masks_on_image(
                (img_src[i, t] * 255).astype(np.uint8),
                tgt_name='',
                alpha=lbl_src[i, t] * 0.5,
                colors=colors[i],
            )
        vis_img[i*H//2:(i+1)*H//2, (T-1)*W//2:T*W//2] = overlay_masks_on_image(
            (img_tgt[i] * 255).astype(np.uint8),
            tgt_name='',
            alpha=lbl_tgt[i] * 0.5,
            colors=colors[i],
        )
        vis_img[i*H//2:(i+1)*H//2, T*W//2:(T+1)*W//2] = overlay_masks_on_image(
            (img_tgt[i] * 255).astype(np.uint8),    
            tgt_name='',
            alpha=pred_tgt[i],
            colors=colors[i],
        )
    img_vis = Image.fromarray(vis_img)
    os.makedirs(f"{os.environ['HOME']}/Downloads/vis_train", exist_ok=True)
    img_vis.save(f"{os.environ['HOME']}/Downloads/vis_train/{iter_num}.jpg")



def vis_train_v2(img_src, lbl_src, img_tgt, lbl_tgt, pred_tgt, iter_num, size_per_img=1.0):
    assert img_src.dim() == 5  # (bs, T, 3, H, W)
    lbl_src = F.interpolate(
            lbl_src.flatten(0, 1).float(),
            img_src.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    bs, nsrc, _, H, W = img_src.shape
    n_notions = lbl_tgt.shape[1]
    lbl_src = lbl_src.view(bs, nsrc, n_notions, H, W)
    T = img_src.shape[1] + 1

    ncol = T * 2 + 1
    _, axs = plt.subplots(nrows=bs, ncols=ncol, figsize=(ncol*size_per_img, bs*size_per_img))

    # unnormalize images
    img_src = img_src * 0.25 + 0.5
    img_tgt = img_tgt * 0.25 + 0.5
    img_src = img_src.detach().permute(0, 1, 3, 4, 2).cpu().numpy()
    img_tgt = img_tgt.detach().permute(0, 2, 3, 1).cpu().numpy()
    lbl_src = lbl_src.detach().cpu().numpy()
    lbl_tgt = lbl_tgt.squeeze(1).detach().cpu().numpy()
    pred_tgt = pred_tgt.detach().sigmoid().squeeze().cpu().numpy()

    for i in range(bs):
        for t in range(T-1):
            axs[i, t*2].imshow(img_src[i, t, :, :])
            axs[i, t*2 + 1].imshow(lbl_src[i, t])
        axs[i, (T-1)*2].imshow(img_tgt[i])
        axs[i, T*2-1].imshow(lbl_tgt[i])
        axs[i, T*2].imshow(pred_tgt[i])
    # set all axes off
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(f"{os.environ['HOME']}/Downloads/vis_train", exist_ok=True)
    plt.savefig(f"{os.environ['HOME']}/Downloads/vis_train/{iter_num}.jpg")
    plt.close()


def vis_inference(img_src, lbl_src, img_tgt, pred_tgt):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))

    axs[0].imshow(img_src[..., ::-1])
    axs[1].imshow(lbl_src)
    axs[2].imshow(img_tgt[..., ::-1])
    axs[3].imshow(pred_tgt)

    plt.tight_layout()
    plt.savefig("/home/yuan/Downloads/vis_inference.png")
    plt.close()


def show_masks(masks):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:,:,3] = 0
    for m in masks:
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def apply_colored_mask(image, mask, color, alpha=0.5):
    """
    Apply a semi-transparent RGB color mask to an image.

    Parameters:
    - image (np.ndarray): Input image (H, W, 3), dtype=uint8
    - mask (np.ndarray): Binary mask (H, W), values are float in range [0, 1] or boolean
    - color (tuple): RGB color as a tuple, e.g., (255, 0, 0)
    - alpha (float or np.ndarray): Transparency factor for the mask overlay

    Returns:
    - np.ndarray: Image with color mask applied to masked region
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Create a copy of the image to overlay on
    output_image = image[..., ::-1].copy()

    if mask.any():
        # Create a color layer of the same shape as the image
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:, :] = color

        # Alpha blending: transparency
        if isinstance(alpha, float):
            assert mask.dtype == bool, "Mask must be boolean when alpha is a float"
            pil_image = Image.fromarray(image)
            pil_color_layer = Image.new("RGB", image.shape[1::-1], color)
            pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            blended = Image.composite(pil_image, pil_color_layer, pil_mask)
            output_image[mask] = np.array(blended)[mask]
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape == mask.shape, "Alpha mask must have the same shape as the binary mask"
            alpha_expanded = np.expand_dims(alpha, axis=-1)  # Shape (H, W, 1)
            output_image[mask] = (image[mask] * (1 - alpha_expanded[mask]) + 
                                  color_layer[mask] * alpha_expanded[mask]).astype(np.uint8)
        else:
            raise ValueError("Alpha must be either a float or a numpy array")

    return output_image


def apply_jet_colormap_transparent(image, mask):
    """
    Apply a transparent Jet colormap to an image using a float mask.

    Parameters:
    - image (np.ndarray): Input image (H, W, 3), RGB, dtype=uint8
    - mask (np.ndarray): Float mask (H, W), values in [0.0, 1.0]

    Returns:
    - np.ndarray: Image with Jet colormap blended over original image
    """
    # Ensure mask is float32 and clipped between 0 and 1
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    # Resize if needed
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width")

    # Convert mask to Jet colormap using matplotlib
    jet_map = plt.get_cmap('jet')
    jet_colors = jet_map(1 - mask)[..., :3] # Ignore alpha, get RGB only
    # jet_colors[..., -1] = 0.6
    jet_colors = (jet_colors * 255).astype(np.uint8)

    # Blend the jet colormap with the original image using the mask as alpha
    blended = image.copy().astype(np.float32)
    jet_colors = jet_colors.astype(np.float32)

    for c in range(3):
        blended[:, :, c] = (1 - 0.5) * blended[:, :, c] + 0.5 * jet_colors[:, :, c]

    return blended.astype(np.uint8)


