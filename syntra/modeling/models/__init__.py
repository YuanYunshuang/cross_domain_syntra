
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple



class SemSegModelBase(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        src_fusion: str = "max",
    ):
        """
        Args:
            src_fusion: how the per-source predictions are merged into one mask.
                "max"  - element-wise maximum over the sources (default, used by all runs so far)
                "mean" - mean over the sources that actually carry the class in their label
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        assert src_fusion in ("max", "mean"), f"Unknown source fusion: {src_fusion}"
        self.src_fusion = src_fusion

    def fuse_sources(self, masks, src_lbl_mask):
        """Merge the per-source mask logits (B, T-1, N, 1, H, W) into (B, N, 1, H, W).

        Sources whose label does not contain the queried class carry no evidence; they are
        pushed to a very low logit for "max" and excluded from the average for "mean".

        Args:
            masks: per-source mask logits, shape (B, T-1, N, 1, H, W).
            src_lbl_mask: bool tensor (B, T-1, N, 1, 1, 1), True where the source label is
                non-empty for that class.
        """
        neg = torch.log(torch.tensor(1e-4, device=masks.device))
        if self.src_fusion == "max":
            masks = torch.where(src_lbl_mask, masks, neg)
            return masks.max(dim=1).values

        # mean over valid sources only; fall back to `neg` when no source carries the class
        valid = src_lbl_mask.to(masks.dtype)
        masked_sum = (masks * valid).sum(dim=1)
        count = valid.sum(dim=1)
        return torch.where(count > 0, masked_sum / count.clamp_min(1.0), neg)

    def trainable_state_dict(self) -> dict:
        filtered_state_dict = {}
        state_dict = self.state_dict()
        for key, value in self.named_parameters():
            if value.requires_grad:
                filtered_state_dict[key] = state_dict[key]
        return filtered_state_dict
    
    def load_trainable_state_dict(self, state_dict: dict) -> None:
        full_state_dict = self.state_dict()
        for key, value in state_dict.items():
            full_state_dict[key] = value
        self.load_state_dict(full_state_dict)

    def forward(self, img_src, lbl_src, img_tgt, aux=None):
        raise NotImplementedError
    

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        # original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size[0], input_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks
