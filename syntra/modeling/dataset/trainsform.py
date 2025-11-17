
import torch
import numpy as np
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


    

class PadToTargetSize:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def pad_one(self, image, label):
        w, h = image.shape[1:]
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
                i = F.hflip(i)
                if l is not None:
                    l = F.hflip(l)
            # vertical flip
            if do_v:
                i = F.vflip(i)
                if l is not None:
                    l = F.vflip(l)

            # rotation: images bilinear, labels nearest
            i = F.rotate(i, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            if l is not None:
                # use nearest to preserve mask values; ensure bool dtype after
                l = F.rotate(l.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST, 
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

