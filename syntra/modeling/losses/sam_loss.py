import torch.nn as nn
import monai
import torch


class SAMLoss(nn.Module):
    """Dice + BCE over the positive and negative branches of a batch.

    This is v3's objective restored. Between v3 (49843ea, 2026-01-25) and the E13 branch the
    loss drifted through three commits and stopped computing what it reports:

      97facdc  `reduction='none'` on both terms, so a per-subset weight could be applied to each
               sample (5.0 for siegfried, 1.0 for the rest).
      da187e4  that per-subset weighting was removed again - but `reduction='none'` stayed, and
               an ignore filter was added on top of it:
                   cared = gt_masks_pos != self.ignore_index
                   pred_masks_pos = pred_masks_pos[cared].unsqueeze(1)
               Boolean-indexing an (M, H, W) tensor with an (M, H, W) mask returns a *flat* 1-D
               tensor, so this reshaped the batch to (P, 1) - every pixel its own sample, no
               spatial dimension left. monai reduces over dims 2.. , and `torch.sum(dim=[])`
               reduces over *everything*, so DiceLoss returned a single pooled scalar instead of
               one value per row. Macro became micro: on a batch of one large and one small
               object, 0.5705 became 0.3753, and small objects stopped contributing. The filter
               itself was inert - `!= 255` on a binary mask is all-True - so it deleted the
               spatial structure to achieve nothing.
      28cf0f6  `train_one_gpu.py` began passing `dice_neg_weight=getattr(args, ..., 1.0)`. The
               call-site fallback is 1.0 and no config sets the key, so this class's own default
               of 0.0 never applied and a Dice term on the all-negative samples was live in every
               run. Measured on an all-negative sample it is a near-constant +1.0 with a gradient
               norm of 1e-10 to 1e-13: it trains nothing and inflates the loss. v3 had no Dice on
               negatives at all, and since `model_best` selects on windowed *training* loss, it
               made "best" partly a function of how many negatives a window happened to hold.

    Together these are the difference between v3's seg_loss, which stays in [0.0099, 0.8666],
    and the E13 branch's, which reaches 2.0000 on 8,906 of 20,000 logged steps - impossible for
    a Dice loss, which is bounded in [0, 1].

    **The Dice here is row-wise, and that is deliberate.** `pred_masks[pos_inds].squeeze(1)` is
    (M, H, W); monai reads dim 1 as the channel dim and reduces only over dim 2, so it returns
    one Dice per (sample, row) - shape (M, H, 1) - and the mean is taken over rows as well as
    samples. Almost certainly not what anyone intended, but it is the objective that produced
    85.89 and 86.57 and it is not being changed here under the guise of a bug fix. Making it
    per-image is a one-line change (unsqueeze both to (M, 1, H, W)) and a separate experiment
    with its own validation run - not something to adopt silently.

    Ignored pixels are handled without touching the shape: the label goes to 0 and the logit to
    -1e4, whose sigmoid is exactly 0 in fp32, so the pixel adds nothing to the intersection, to
    `pred_o` or to `ground_o` and carries no gradient. The v3 corpus contains no 255 at all, so
    on it this is bit-for-bit v3; the current corpus does, which is why the filter has to work
    rather than merely be deleted.
    """

    def __init__(self,
                 ignore_index=255,
                 ce_neg_weight=1.0,
                 dice_pos_weight=1.0,
                 dice_neg_weight=0.0,
                 **kwargs):
        super(SAMLoss, self).__init__()
        self.seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.ignore_index = ignore_index
        self.ce_neg_weight = ce_neg_weight
        self.dice_pos_weight = dice_pos_weight
        self.dice_neg_weight = dice_neg_weight

    def _masked(self, pred, gt, valid):
        """Neutralise the ignored pixels without collapsing the spatial dimensions."""
        gt = torch.where(valid, gt, torch.zeros_like(gt))
        pred = torch.where(valid, pred, torch.full_like(pred, -1e4))
        return pred, gt

    def forward(self, pred_masks, gt_masks, **kwargs):
        """
        Args:
            pred_masks: (B, N, 1, H, W) predicted masks before sigmoid
            gt_masks: (B, N, H, W) ground truth masks
        Returns:
            total_loss: scalar tensor
        """
        gt = gt_masks.float()
        valid = gt != self.ignore_index                       # (B, N, H, W)
        # A target counts as positive only if it carries foreground that was actually annotated.
        # v3 used `gt_masks.sum(-1) > 0`, which on a corpus with 255 in it counts ignored pixels
        # as evidence of the class and routes an entirely-ignored mask into the positive branch.
        # On the v3 corpus, which has no 255, the two agree exactly.
        pos_inds = ((gt == 1) & valid).flatten(2, 3).any(-1)  # (B, N)

        pred_pos, gt_pos = self._masked(pred_masks[pos_inds].squeeze(1),
                                        gt[pos_inds], valid[pos_inds])
        pred_neg, gt_neg = self._masked(pred_masks[~pos_inds].squeeze(1),
                                        gt[~pos_inds], valid[~pos_inds])
        valid_pos, valid_neg = valid[pos_inds], valid[~pos_inds]

        # A batch can hold no positive at all - every target a class the tile does not carry -
        # and `.mean()` of an empty tensor is NaN, not zero. Guard on there being a valid pixel
        # rather than on the sample count, since a mask can be entirely ignored.
        zero = torch.zeros((), device=pred_masks.device)
        has_pos = valid_pos.any()
        has_neg = valid_neg.any()

        def masked_bce(pred, gt, valid):
            return (self.ce_loss_fn(pred, gt) * valid).sum() / valid.sum().clamp(min=1)

        ce_loss_pos = masked_bce(pred_pos, gt_pos, valid_pos) if has_pos else zero
        ce_loss_neg = (self.ce_neg_weight * masked_bce(pred_neg, gt_neg, valid_neg)
                       if has_neg else zero)

        # The weights multiply. They used to be read as gates - `if self.dice_pos_weight > 0` and
        # then the term added at full strength - so 0.5 meant 1.0 and only 0.0 meant anything.
        seg_loss_pos = (self.dice_pos_weight * self.seg_loss_fn(pred_pos, gt_pos)
                        if self.dice_pos_weight and has_pos else zero)
        seg_loss_neg = (self.dice_neg_weight * self.seg_loss_fn(pred_neg, gt_neg)
                        if self.dice_neg_weight and has_neg else zero)

        total_loss = seg_loss_pos + seg_loss_neg + ce_loss_pos + ce_loss_neg

        return total_loss, {"seg_loss": seg_loss_pos.item() + seg_loss_neg.item(),
                            "ce_loss_pos": ce_loss_pos.item(),
                            "ce_loss_neg": ce_loss_neg.item()}
