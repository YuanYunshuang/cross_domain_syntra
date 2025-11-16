import torch.nn as nn
import monai
import torch


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()
        self.seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.ignore_index = 255

    def forward(self, pred_masks, gt_masks, **kwargs):
        if 'aux' in kwargs and kwargs['aux']:
            valid_notions_inds = []
            for i, aux in enumerate(kwargs['aux']):
                n_cls = len(aux['cls_name'])
                valid_notions_inds.extend([(i, j) for j in range(n_cls)])
        else:
            assert gt_masks.shape[1] == 1, \
                "During training, aux info must be provided if multiple notions is used." \
                "The formats of aux info should be a list of dict, each dict contains 'cls_name' key."
            valid_notions_inds = [(i, 0) for i in range(gt_masks.shape[0])]
        gt_masks = gt_masks.unsqueeze(-3)

        b_inds, n_inds = zip(*valid_notions_inds)
        pred_masks = pred_masks[list(b_inds), list(n_inds), ...]
        gt_masks = gt_masks[list(b_inds), list(n_inds), ...]
        ignore = gt_masks == self.ignore_index
        cared = torch.logical_not(ignore)
        pred_masks[ignore] = 0.0
        gt_masks[ignore] = 0
        seg_loss = self.seg_loss_fn(pred_masks, gt_masks)
        ce_loss = self.ce_loss_fn(pred_masks[cared], gt_masks[cared].float())

        total_loss = seg_loss + ce_loss
        return total_loss