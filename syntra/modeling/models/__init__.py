
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
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

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

    def forward(self, img_src, lbl_src, img_tgt):
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
