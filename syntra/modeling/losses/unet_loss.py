import torch.nn as nn
import monai
import torch


class UNETloss(nn.Module):
    def __init__(self, ignore_index=255):
        super(UNETloss, self).__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
        self.ignore_index = ignore_index

    def forward(self, pred_masks, gt_masks, **kwargs):
        """
        Args:
            pred_masks: (B, C, H, W) predicted masks before sigmoid
            gt_masks: (B, H, W) ground truth masks
        Returns:
            total_loss: scalar tensor
        """
        seg_loss = self.ce_loss_fn(pred_masks, gt_masks.squeeze(1).long())
        return seg_loss, {}