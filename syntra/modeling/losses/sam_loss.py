import torch.nn as nn
import monai
import torch


class SAMLoss(nn.Module):
    def __init__(self, 
                 ignore_index=255, 
                 ce_neg_weight=1.0, 
                 dice_pos_weight=1.0,
                 dice_neg_weight=0.0,
                 **kwargs):
        super(SAMLoss, self).__init__()
        self.seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='none')
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.ignore_index = ignore_index
        self.ce_neg_weight = ce_neg_weight
        self.dice_pos_weight = dice_pos_weight
        self.dice_neg_weight = dice_neg_weight

    def forward(self, pred_masks, gt_masks, **kwargs):
        """
        Args:
            pred_masks: (B, N, 1, H, W) predicted masks before sigmoid
            gt_masks: (B, N, H, W) ground truth masks
        Returns:
            total_loss: scalar tensor
        """
        pos_inds = gt_masks.flatten(2, 3).sum(-1) > 0  # (B, N)
        # weights = self.get_dataset_weights(kwargs['aux'], pred_masks.device) # B
        # weights = weights.unsqueeze(1).expand(-1, gt_masks.shape[1])  # (B, N)
        # dice loss only considers the foreground class
        pred_masks_pos = pred_masks[pos_inds].squeeze(1)  # (M, 1, H, W)
        gt_masks_pos = gt_masks[pos_inds].float()  # (M, H, W)
        
        cared = gt_masks_pos != self.ignore_index
        pred_masks_pos = pred_masks_pos[cared].unsqueeze(1)
        gt_masks_pos = gt_masks_pos[cared].unsqueeze(1)

        pred_masks_neg = pred_masks[~pos_inds].squeeze(1)  # (K, 1, H, W)
        gt_masks_neg = gt_masks[~pos_inds].float()  # (K, H, W)
        cared_neg = gt_masks_neg != self.ignore_index
        pred_masks_neg = pred_masks_neg[cared_neg].unsqueeze(1)
        gt_masks_neg = gt_masks_neg[cared_neg].unsqueeze(1)

        
        # A batch can hold no positive at all - every target a class the tile does not carry -
        # and `.mean()` of an empty tensor is NaN, not zero. The negative branch was guarded and
        # the positive one was not, so any all-negative batch poisoned the loss. The old runs
        # never hit it often enough to notice; a budget that buys negatives explicitly does.
        # Both branches are guarded on the tensor being non-empty rather than on the index
        # count, because `cared` can empty one after the split when a mask is entirely ignored.
        zero = torch.zeros((), device=pred_masks.device)
        has_pos = pred_masks_pos.numel() > 0
        has_neg = pred_masks_neg.numel() > 0

        ce_loss_pos = self.ce_loss_fn(pred_masks_pos, gt_masks_pos).mean() if has_pos else zero
        if has_neg:
            ce_loss_neg = self.ce_loss_fn(pred_masks_neg, gt_masks_neg).mean() * self.ce_neg_weight
        else:
            ce_loss_neg = zero

        if self.dice_pos_weight > 0 and has_pos:
            seg_loss_pos = self.seg_loss_fn(pred_masks_pos, gt_masks_pos).mean()
        else:
            seg_loss_pos = zero
        if self.dice_neg_weight > 0 and has_neg:
            seg_loss_neg = self.seg_loss_fn(pred_masks_neg, gt_masks_neg).mean()
        else:
            seg_loss_neg = zero

        # if 'aux' in kwargs and kwargs['aux']:
        #     valid_notions_inds = []
        #     for i, aux in enumerate(kwargs['aux']):
        #         n_cls = len(aux['cls_name'])
        #         valid_notions_inds.extend([(i, j) for j in range(n_cls)])
        # else:
        #     assert gt_masks.shape[1] == 1, \
        #         "During training, aux info must be provided if multiple notions is used." \
        #         "The formats of aux info should be a list of dict, each dict contains 'cls_name' key."
        #     valid_notions_inds = [(i, 0) for i in range(gt_masks.shape[0])]
        # gt_masks = gt_masks.unsqueeze(-3)

        # b_inds, n_inds = zip(*valid_notions_inds)
        # pred_masks = pred_masks[list(b_inds), list(n_inds), ...]
        # gt_masks = gt_masks[list(b_inds), list(n_inds), ...]
        # ignore = gt_masks == self.ignore_index
        # cared = torch.logical_not(ignore)
        # pred_masks[ignore] = 0.0
        # gt_masks[ignore] = 0
        # seg_loss = self.seg_loss_fn(pred_masks, gt_masks)
        # ce_loss = self.ce_loss_fn(pred_masks[cared], gt_masks[cared].float())

        total_loss = seg_loss_pos + seg_loss_neg + ce_loss_pos + ce_loss_neg
            
        return total_loss, {"seg_loss": seg_loss_pos.item() + seg_loss_neg.item(),
                            "ce_loss_pos": ce_loss_pos.item(),
                            "ce_loss_neg": ce_loss_neg.item()}