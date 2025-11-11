
import os
import urllib
import urllib.request

import torch
from .hieradet import Hiera
from .image_encoder import ImageEncoder, FpnNeck
from .position_encoding import PositionEmbeddingSine



def build_sam2_img_encoder(model_type: str, pretrained=True, out_chans=256, **kwargs):
    model_params = {
        "hiera_bp": {
            "trunk": {
                "embed_dim": 112,
                "num_heads": 2,
                "drop_path_rate": 0.1,
            },
            "neck": {
                "backbone_channel_list": [896, 448, 224, 112],
                "fpn_top_down_levels": [2, 3],
            }
        },
    }
    assert model_type in model_params, f"Unknown model type: {model_type}"
    parames = model_params[model_type]
    trunk = Hiera(
        embed_dim=parames["trunk"]["embed_dim"],
        num_heads=parames["trunk"]["num_heads"],
        drop_path_rate=parames["trunk"]["drop_path_rate"],
    )
    neck = FpnNeck(
        backbone_channel_list=parames["neck"]["backbone_channel_list"],
        d_model=out_chans,
        fpn_top_down_levels=parames["neck"]["fpn_top_down_levels"],
        fpn_interp_model='nearest',
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=out_chans,
        )
    )
    image_encoder = ImageEncoder(
        scalp=1,
        trunk=trunk,
        neck=neck,
    )

    if pretrained:
        ckpt_map = {
            "hiera_bp": "sam2.1_hiera_base_plus.pt",
            "hiera_t": "sam2.1_hiera_tiny.pt",
            "hiera_s": "sam2.1_hiera_small.pt",
            "hiera_l": "sam2.1_hiera_large.pt",
        }
        checkpoint_path = f"./checkpoints/{ckpt_map[model_type]}"
        if not os.path.exists(checkpoint_path):
            SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
            url_path = f"{SAM2p1_BASE_URL}/{ckpt_map[model_type]}"
            urllib.request.urlretrieve(url_path, checkpoint_path)
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {k.replace('image_encoder.', '') : v for k, v in state_dict['model'].items() if "image_encoder" in k}
        image_encoder.load_state_dict(state_dict, strict=True)

    return image_encoder