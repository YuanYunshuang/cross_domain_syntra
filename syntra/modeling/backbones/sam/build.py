# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .image_encoder import ImageEncoderViT


def build_sam_img_encoder(mtype, pretrained=True, out_chans=256, **kwargs):
    sam_img_encoder_params = {
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
            "checkpoint": "sam_vit_b_01ec64.pth",
        },
        "vit_l": {
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [5, 11, 17, 23],
            "checkpoint": "sam_vit_l_0b3195.pth",
        },
        "vit_h": {
            "encoder_embed_dim": 1280,
            "encoder_depth": 32,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [7, 15, 23, 31],
            "checkpoint": "sam_vit_h_4b8939.pth",
        },
    }
    assert mtype in sam_img_encoder_params, f"Unknown model type: {mtype}"
    params = sam_img_encoder_params[mtype]
    image_encoder=ImageEncoderViT(
            depth=params["encoder_depth"],
            embed_dim=params["encoder_embed_dim"],
            img_size=1024,  
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=params["encoder_num_heads"],
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=params["encoder_global_attn_indexes"],
            window_size=14,
            out_chans=out_chans,
        )
    if not pretrained:
        return image_encoder

    checkpoint = Path(f"checkpoints/{params['checkpoint']}") 
    if not checkpoint.exists():
        cmd = input(f"Download {checkpoint.name} from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {checkpoint.name} checkpoint...")
            url_map = {
                "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            }
            urllib.request.urlretrieve(
                url_map[checkpoint.name],
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.")
        }
        image_encoder.load_state_dict(state_dict, strict=True)

    return image_encoder
        

if __name__ == "__main__":
    encoder = build_sam_img_encoder("vit_b", pretrained=True)
    print(encoder)


