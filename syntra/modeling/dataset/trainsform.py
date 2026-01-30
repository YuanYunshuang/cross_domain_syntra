
import math
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
from typing import Iterable


    

class Pad:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def pad_one(self, image, label):
        h, w = image.shape[1:]
        assert h <= self.target_size and w <= self.target_size, \
            f"Image size ({w}, {h}) larger than target size {self.target_size}."
        pad_w = max(self.target_size - w, 0)
        pad_h = max(self.target_size - h, 0)

        # Compute padding (left, top, right, bottom)
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        # Pad image with 0
        image = F.pad(image, [left, top, right, bottom], fill=0)
        # Pad label with 255
        if label is not None:
            label = F.pad(label, [left, top, right, bottom], fill=False)
        return image, label

    def __call__(self, image, label):
        if isinstance(image, list):
            image_list = []
            label_list = []
            for img, lbl in zip(image, label):
                img, lbl = self.pad_one(img, lbl)
                image_list.append(img)
                label_list.append(lbl)
            return image_list, label_list
        else:
            image, label = self.pad_one(image, label)

        return image, label


class Resize:
    def __init__(self, target_size):
        self.target_size = target_size
        self.resize_img = v2.Resize((target_size, target_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_lbl = v2.Resize((target_size, target_size), interpolation=InterpolationMode.NEAREST)
    
    def resize(self, imgs, lbls):
        """
        Resize images and labels to target size. Supports either a single image/label
        or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        def apply_one(i, l):
            i = self.resize_img(i)
            if l is not None:
                l = self.resize_lbl(l.unsqueeze(0)).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_one(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_one(imgs, lbls)
    
    def __call__(self, img, lbl):
        return self.resize(img, lbl)
    

class Flip:
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, imgs, lbls):
        """
        Apply random horizontal and/or vertical flips to images and labels.
        Supports either a single image/label or lists of them.
        Sample once per call so all items get identical transform.

        Args:
            imgs: single image or list of images
            lbls: single label or list of labels
        Returns:
            Transformed images and labels.
        """
        do_h = self.horizontal and (np.random.rand() < 0.5)
        do_v = self.vertical and (np.random.rand() < 0.5)

        def apply_flip(img, lbl):
            if do_h:
                img = F.hflip(img)
                if lbl is not None:
                    lbl = F.hflip(lbl)
            if do_v:
                img = F.vflip(img)
                if lbl is not None:
                    lbl = F.vflip(lbl)
            return img, lbl

        if isinstance(imgs, list):
            return zip(*[apply_flip(i, l) for i, l in zip(imgs, lbls)])
        else:
            return apply_flip(imgs, lbls)
        

class ToTensor:
    def __init__(self, img_dtype=torch.float32, lbl_dtype=torch.uint8, img_scale=True, lbl_scale=False):
        self.img_dtype = img_dtype
        self.lbl_dtype = lbl_dtype
        self.img_scale = img_scale
        self.lbl_scale = lbl_scale

        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(img_dtype, scale=True),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(lbl_dtype, scale=False),
        ])

    def __call__(self, imgs, lbls=None):
        img = self.img_transforms(imgs)
        lbl = self.lbl_transforms(lbls) if lbls is not None else None
        return img, lbl


class Normalize:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
        self.mean = mean
        self.std = std
        self.normalize = v2.Normalize(mean=mean, std=std)

    def __call__(self, imgs, lbls):
        img = self.normalize(imgs)
        return img, lbls
    

class Rotate:
    def __init__(self, degrees=30):
        self.degrees = degrees
    
    def __call__(self, imgs, lbls):
        angle = float(np.random.uniform(-self.degrees, self.degrees))

        def apply_rotation(i, l):
            i = F.rotate(i, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            l = F.rotate(l.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST, 
                             expand=False, fill=False).squeeze(0)
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_rotation(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_rotation(imgs, lbls)

class RandomRotateBatch:
    def __init__(self, degrees=30):
        assert degrees >= 0, "degrees must be non-negative."
        self.degrees = abs(degrees)

    def _generate_rotation_matrix(self, images):
        """
        Generate random rotation matrices for a batch.
        Args:
            images: Tensor of shape (B, C, H, W)
        """
        # Degree range
        min_deg, max_deg = -self.degrees, self.degrees

        # Sample one angle per (image, label) pair
        B = images.shape[0]
        angles = (min_deg - max_deg) * torch.rand(B, device=images.device) + max_deg
        angles = angles.to(dtype=images.dtype)

        # Build rotation matrices
        radians = angles * math.pi / 180.0
        cos = torch.cos(radians)
        sin = torch.sin(radians)

        theta = torch.zeros((B, 2, 3), device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        return theta

    def __call__(self, imgs, lbls):
        assert imgs.dim() in [4, 5], "images must be (B, T, C, H, W), or (B, C, H, W)"
        assert lbls.dim() in [3, 4, 5], "labels must be (B, T, H, W) or (B, T, 1, H, W) or (B, H, W) or (B, 1, H, W)"

        lbl_shape = lbls.shape
        img_shape = imgs.shape
        if imgs.dim() == 4:
            B, C, H, W = imgs.shape
            T = 1
        else:
            B, T, C, H, W = imgs.shape
            # Make sure images are 4D for grid_sample
            imgs = imgs.flatten(0, 1)  
            lbls = lbls.flatten(0, 1) 
        
        # Make sure labels are 4D for grid_sample
        squeeze_labels = False
        if lbls.dim() == 3:
            lbls = lbls.unsqueeze(1)  # (B, 1, H, W)
            squeeze_labels = True
        else:
            assert tuple(lbls.shape) == (imgs.shape[0], 1, H, W), \
                "labels dimension must match images, either (B, 1, H, W) or (B, H, W)"
        # Generate rotation matrices
        theta = self._generate_rotation_matrix(imgs)
        # Create sampling grid
        grid = F.affine_grid(theta, imgs.size(), align_corners=False)
        # Rotate images
        rotated_imgs = F.grid_sample(
            imgs, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )
        # Rotate labels
        rotated_lbls = F.grid_sample(
            lbls.to(dtype=imgs.dtype), grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False
        )
        if squeeze_labels:
            rotated_lbls = rotated_lbls.squeeze(1)
        return rotated_imgs.reshape(*img_shape), rotated_lbls.reshape(*lbl_shape)


class RandomFlipBatch:
    def __init__(self, p_horizontal=0.5, p_vertical=0.0):
        """
        Randomly flip images and labels in a batch.

        Args:
            p_horizontal: probability of horizontal flip per (image, label) pair
            p_vertical: probability of vertical flip per (image, label) pair
        """
        assert 0.0 <= p_horizontal <= 1.0, "p_horizontal must be in [0, 1]."
        assert 0.0 <= p_vertical <= 1.0, "p_vertical must be in [0, 1]."
        self.p_horizontal = float(p_horizontal)
        self.p_vertical = float(p_vertical)

    def __call__(self, imgs, lbls):
        """
        Args:
            imgs: Tensor of shape (B, C, H, W)
            lbls: Tensor of shape (B, H, W) or (B, 1, H, W)

        Returns:
            flipped_imgs, flipped_lbls
        """
        assert imgs.dim() == 5, "images must be (B, T, C, H, W)"
        assert lbls.dim() in [4, 5], "labels must be (B, T, H, W) or (B, T, 1, H, W)"
        B, T, C, H, W = imgs.shape
        lbl_shape = lbls.shape

        # Make sure images are 4D for grid_sample
        imgs = imgs.flatten(0, 1)  
        lbls = lbls.flatten(0, 1) 
        # TODO: try using the same rotation for all T images in one batch 

        # Ensure labels are 4D: (B, 1, H, W)
        squeeze_labels = False
        if lbls.dim() == 3:
            lbls = lbls.unsqueeze(1)
            squeeze_labels = True
        else:
            assert tuple(lbls.shape) == (imgs.shape[0], 1, H, W), \
                "labels dimension must match images, either (B, 1, H, W) or (B, H, W)"

        device = imgs.device

        # --- Random horizontal flips ---
        if self.p_horizontal > 0.0:
            # mask_h[i] == True -> flip sample i horizontally
            mask_h = (torch.rand(B*T, device=device) < self.p_horizontal)

            if mask_h.any():
                # Flip all in batch, then select per-sample via where (no Python loop)
                imgs_h = torch.flip(imgs, dims=[-1])   # flip width
                lbls_h = torch.flip(lbls, dims=[-1])

                mask_h_ = mask_h.view(B*T, 1, 1, 1)      # broadcast over C, H, W
                imgs = torch.where(mask_h_, imgs_h, imgs)
                lbls = torch.where(mask_h_, lbls_h, lbls)

        # --- Random vertical flips ---
        if self.p_vertical > 0.0:
            # mask_v[i] == True -> flip sample i vertically
            mask_v = (torch.rand(B*T, device=device) < self.p_vertical)

            if mask_v.any():
                imgs_v = torch.flip(imgs, dims=[-2])   # flip height
                lbls_v = torch.flip(lbls, dims=[-2])

                mask_v_ = mask_v.view(B*T, 1, 1, 1)
                imgs = torch.where(mask_v_, imgs_v, imgs)
                lbls = torch.where(mask_v_, lbls_v, lbls)

        # Restore original label shape if needed
        if squeeze_labels:
            lbls = lbls.squeeze(1)

        return imgs.reshape(B, T, C, H, W), lbls.reshape(*lbl_shape)
    

class RandomChannelSwapBatch:
    def __init__(self, p=0.5):
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]."
        self.p = p

    def __call__(self, imgs, lbls):
        """
        Args:
            imgs: Tensor of shape (B, T, C, H, W)
            lbls: Tensor of shape (B, T, H, W) or (B, T, 1, H, W)

        Returns:
            swapped_imgs, lbls
        """
        assert imgs.dim() == 5, "images must be (B, T, C, H, W)"
        assert lbls.dim() in [4, 5], "labels must be (B, T, H, W) or (B, T, 1, H, W)"
        B, T, C, H, W = imgs.shape
        device = imgs.device
        if C < 2 or self.p == 0.0:
            return imgs, lbls
        # mask_swap[i] == True -> swap channels for sample i
        mask_swap = (torch.rand(B, device=device) < self.p)
        if mask_swap.any():
            # Generate random permutations for each sample to be swapped
            permuted_imgs = imgs.clone()
            for i in range(B):
                if mask_swap[i]:
                    perm = torch.randperm(C)
                    permuted_imgs[i] = imgs[i, :, perm, :, :]

            imgs = permuted_imgs

        return imgs, lbls
    

class RandomColorJitterBatch:
    def __init__(self, brightness=0.0, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, imgs, lbls):
        """
        Args:
            imgs: Tensor of shape (B, T, C, H, W)
            lbls: Tensor of shape (B, T, H, W) or (B, T, 1, H, W)

        Returns:
            jittered_imgs, lbls
        """
        assert imgs.dim() == 5, "images must be (B, T, C, H, W)"
        assert lbls.dim() in [4, 5], "labels must be (B, T, H, W) or (B, T, 1, H, W)"
        B, T, C, H, W = imgs.shape

        # Apply color jitter per image in batch
        jittered_imgs = []
        for i in range(B):
            img = imgs[i].reshape(-1, C, H, W)  # (T, C, H, W)
            img_jittered = self.color_jitter(img)
            jittered_imgs.append(img_jittered)
        jittered_imgs = torch.stack(jittered_imgs, dim=0)  # (B, T, C, H, W)

        return jittered_imgs, lbls


class TransformCompose:
    def __init__(self, transforms_cfg):
        assert isinstance(transforms_cfg, Iterable), "transforms_cfg must be a list, each item contains key 'name' and 'args'."
        transforms = []
        for t_cfg in transforms_cfg:
            t_name = t_cfg['name']
            t_args = t_cfg.get('args', {})
            if globals().get(t_name) is not None:
                t_class = globals()[t_name]
                transforms.append(t_class(**t_args))
            else:
                raise ValueError(f"Unknown transform name: {t_name}")
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


class MapTransformTrain:
    def __init__(self, image_size=384, rot_degrees=30, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
        # store params for reproducible geometry sampling per pair
        self.rot_degrees = rot_degrees
        self.img_size = image_size

        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bool, scale=False),
        ])

        # self.pad_transforms = v2.Compose([
        #     PadToTargetSize(image_size),
        # ])

        self.resize_img = v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_lbl = v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
    
    def resize(self, imgs, lbls):
        """
        Resize images and labels to target size. Supports either a single image/label
        or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        def apply_one(i, l):
            i = self.resize_img(i)
            if l is not None:
                l = self.resize_lbl(l.unsqueeze(0)).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_one(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_one(imgs, lbls)


    def geometric_transforms(self, img, lbl):
        """
        Sample a single geometric transform (hflip, vflip, rotation angle)
        and apply the same transform to all images and labels. Supports
        either a single image/label or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        # sample once per call so all items get identical transform
        do_h = np.random.rand() < 0.5
        do_v = np.random.rand() < 0.5
        angle = float(np.random.uniform(-self.rot_degrees, self.rot_degrees))

        def apply_one(i, l):
            # horizontal flip
            if do_h:
                i = TF.hflip(i)
                if l is not None:
                    l = TF.hflip(l)
            # vertical flip
            if do_v:
                i = TF.vflip(i)
                if l is not None:
                    l = TF.vflip(l)

            # rotation: images bilinear, labels nearest
            i = TF.rotate(i, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            if l is not None:
                # use nearest to preserve mask values; ensure bool dtype after
                l = TF.rotate(l.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST, 
                             expand=False, fill=False).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(img, list):
            imgs = []
            lbls = []
            for i, ll in zip(img, lbl):
                ii, ll = apply_one(i, ll)
                imgs.append(ii)
                lbls.append(ll)
            return imgs, lbls
        else:
            return apply_one(img, lbl)

    def __call__(self, img, lbl):
        img = self.img_transforms(img)
        lbl = self.lbl_transforms(lbl)
        
        img, lbl = self.resize(img, lbl)
        img, lbl = self.geometric_transforms(img, lbl)

        if isinstance(lbl, list):
            lbl = [l for l in lbl]
        else:
            lbl = lbl
        return img, lbl
    

class MapTransformTest:
    def __init__(self, image_size=384, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.bool, scale=False),
        ])

        # self.pad_transforms = v2.Compose([
        #     PadToTargetSize(image_size),
        # ])

        self.resize_img = v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_lbl = v2.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
    
    def resize(self, imgs, lbls):
        """
        Resize images and labels to target size. Supports either a single image/label
        or lists of them.
        Images use bilinear interpolation and labels use nearest.
        """
        def apply_one(i, l):
            i = self.resize_img(i)
            if l is not None:
                l = self.resize_lbl(l.unsqueeze(0)).squeeze(0)
                try:
                    l = l.to(torch.bool)
                except Exception:
                    # fallback if returned type isn't a tensor
                    pass
            return i, l

        if isinstance(imgs, list):
            img_list = []
            lbl_list = []
            for i, ll in zip(imgs, lbls):
                ii, ll = apply_one(i, ll)
                img_list.append(ii)
                lbl_list.append(ll)
            return img_list, lbl_list
        else:
            return apply_one(imgs, lbls)

    def __call__(self, img, lbl=None):
        img = self.img_transforms(img)
        if lbl is not None:
            lbl = self.lbl_transforms(lbl)
            img, lbl = self.resize(img, lbl)
            return img, lbl
        else:
            img, lbl = self.resize(img, lbl)
            return img

