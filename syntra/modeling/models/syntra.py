
import torch
from torch.nn import functional as F

from . import SemSegModelBase


class SyntraModel(SemSegModelBase):
    def forward(self, img_src, lbl_src, img_tgt):
        image = torch.cat([img_tgt.unsqueeze(1), img_src], dim=1) # (B, T, 3, H, W)
        B, T, _, H, W = image.shape
        N = lbl_src.shape[2]
        backbone_out = self.image_encoder(image.flatten(0, 1))  # (B*T, 256, H//8, W//8)
        backbone_out = self.reshape_backbone_out(backbone_out, B, T)

        image_embedding = backbone_out['vision_features'] # (B, T, 256, H//8, W//8)
        tgt_img_embedding = image_embedding[:, :1].repeat(1, (T-1)*N, 1, 1, 1) # (B, (T-1)*N, 256, H//8, W//8)
        src_img_embedding = image_embedding[:, 1:].unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # (B, T-1, N, 256, H//8, W//8)

        # prepare high resolution features
        high_res_features = [x[:, 0] for x in backbone_out['backbone_fpn'][:2]]
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            imgs_emb=src_img_embedding.flatten(0, 2),  # (B*(T-1)*N, 256, H//8, W//8)
            masks=lbl_src.flatten(0, 2).unsqueeze(1).float(), # (B*(T-1)*N, 1, H, W
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=tgt_img_embedding.flatten(0, 1),  # (B, (T-1)*N, 256, H//8, W//8) -> (B*(T-1)*N, 256, H//8, W//8)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, H//8, W//8)
            sparse_prompt_embeddings=sparse_embeddings,  # (B*(T-1)*N, N_sparse, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B*(T-1)*N, 256, H//8, W//8)
            high_res_features=high_res_features,  # [(B*(T-1)*N, C, H_high, W_high), ...]
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,  # (B*(T-1)*N, 1, 96, 96)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, T-1, N, 1, H, W)  # (B, N, 1, H, W)

        src_lbl_mask = lbl_src.flatten(3).any(dim=3).view(B, T-1, N, 1, 1, 1)  # (B, T-1, N)
        neg = torch.log(torch.tensor(1e-4, device=ori_res_masks.device))
        # set the masks to a very low value where there is no source label
        x, y, z = torch.where(src_lbl_mask == False)[:3]
        ori_res_masks[x, y, z] = neg # (B, T-1, N, H, W)
        ori_res_masks = ori_res_masks.max(dim=1).values  # (B, N, 1, H, W)
        return ori_res_masks
    
    def reshape_backbone_out(self, backbone_out, B, T):
        """Reshape the backbone output from (B*T, C, H, W) to (B, T, C, H, W)"""
        return {k: [x.view(B, T, *x.shape[1:]) for x in v] if isinstance(v, list) \
                else v.view(B, T, *v.shape[1:]) for k, v in backbone_out.items()}