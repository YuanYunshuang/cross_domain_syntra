import torch.nn as nn
import monai
import torch


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()
        self.seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.ignore_index = 255

    def forward(self, pred_masks, gt_masks):
        gt_masks = gt_masks.unsqueeze(-3)
        ignore = gt_masks == self.ignore_index
        cared = torch.logical_not(ignore)
        pred_masks[ignore] = 0.0
        gt_masks[ignore] = 0
        seg_loss = self.seg_loss_fn(pred_masks, gt_masks)
        ce_loss = self.ce_loss_fn(pred_masks[cared], gt_masks[cared].float())

        total_loss = seg_loss + ce_loss
        return total_loss