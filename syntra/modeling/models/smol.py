
import torch
from torch.nn import functional as F

from . import SemSegModelBase


class SMOLModel(SemSegModelBase):
    
    def forward(self, img_src, lbl_src, img_tgt, aux=None):
         # img_src: (B, T-1, 3, H, W)
         # lbl_src: (B, T-1, N, H, W)
         # img_tgt: (B, 3, H, W)
        image = torch.cat([img_tgt.unsqueeze(1), img_src], dim=1) # (B, T, 3, H, W)
        B, T, _, H, W = image.shape
        N = lbl_src.shape[2]
        image_embedding = self.image_encoder(image.flatten(0, 1))  # (B*T, 256, 64, 64)
        image_embedding = image_embedding.view(B, T, *image_embedding.shape[1:])  # (B, T, 256, 64, 64)
        tgt_img_embedding = image_embedding[:, :1].repeat(1, (T-1)*N, 1, 1, 1) # (B, (T-1)*N, 256, 64, 64)
        src_img_embedding = image_embedding[:, 1:].unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # (B, T-1, N, 256, 64, 64)
        # do not compute gradients for prompt encoder
        # with torch.no_grad():
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            imgs_emb=src_img_embedding.flatten(0, 2),  # (B*(T-1)*N, 256, 64, 64)
            masks=lbl_src.flatten(0, 2).unsqueeze(1).float(), # (B*(T-1)*N, 1, H, W
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=tgt_img_embedding.flatten(0, 1),  # (B, (T-1)*N, 256, 64, 64) -> (B*(T-1)*N, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 24, 24)
            sparse_prompt_embeddings=sparse_embeddings,  # (B*(T-1)*N, N_sparse, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B*(T-1)*N, 256, 64, 64)
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,  # (B*(T-1)*N, 1, 96, 96)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, T-1, N, 1, H, W)  # (B, N, 1, H, W)

        src_lbl_mask = lbl_src.flatten(3).any(dim=3).view(B, T-1, N, 1, 1, 1)  # (B, T-1, N)
        return self.fuse_sources(ori_res_masks, src_lbl_mask)  # (B, N, 1, H, W)