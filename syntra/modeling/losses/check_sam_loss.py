"""SAMLoss must equal v3's loss on data without an ignore label.

v3 is the run that scored 85.89 and reproduced at 86.57, so it is the definition of correct
here. These are the P0.3 regression checks for the loss: the drift that cost the E13 branch its
accuracy was silent in every log except the magnitude of `seg_loss`, which nobody was reading.

Named `check_` rather than `test_` because in this repo `test_*.py` means an evaluation script
(`syntra/test.py`, `syntra/test_unet.py`). No pytest: neither the `syntra` nor the `hismap` env
has it, and installing it into `syntra` risks moving numpy under torch 2.5.1+cu118.

Run: PYTHONPATH=. python syntra/modeling/losses/check_sam_loss.py
"""
import sys

import monai
import torch
import torch.nn as nn

from syntra.modeling.losses.sam_loss import SAMLoss


class V3SAMLoss(nn.Module):
    """`sam_loss.py` exactly as it stood at 49843ea, the commit that produced 85.89."""

    def __init__(self, ignore_index=255, ce_neg_weight=1.0):
        super().__init__()
        self.seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.ignore_index = ignore_index
        self.ce_neg_weight = ce_neg_weight

    def forward(self, pred_masks, gt_masks, **kwargs):
        pos_inds = gt_masks.flatten(2, 3).sum(-1) > 0
        pred_masks_pos = pred_masks[pos_inds].squeeze(1)
        gt_masks_pos = gt_masks[pos_inds].float()
        pred_masks_neg = pred_masks[~pos_inds].squeeze(1)
        gt_masks_neg = gt_masks[~pos_inds].float()
        seg_loss = self.seg_loss_fn(pred_masks_pos, gt_masks_pos)
        ce_loss_pos = self.ce_loss_fn(pred_masks_pos, gt_masks_pos)
        ce_loss_neg = self.ce_loss_fn(pred_masks_neg, gt_masks_neg)
        return seg_loss + ce_loss_pos + self.ce_neg_weight * ce_loss_neg


def make_batch(seed, B=8, N=1, H=32, W=32, all_negative=False, with_ignore=False):
    """(B, N, 1, H, W) logits and (B, N, H, W) labels, some targets positive, some not."""
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, N, 1, H, W, generator=g)
    gt = torch.zeros(B, N, H, W)
    if not all_negative:
        for b in range(0, B, 2):                      # half the targets carry the class
            gt[b, :, : H // (2 + b % 3), :] = 1
    if with_ignore:
        gt[:, :, -4:, :] = 255
    return pred, gt


def check_matches_v3_without_ignore():
    """The v3 corpus has no 255 in it, so on that data the two must agree to fp32 precision."""
    for seed in (0, 1, 2, 3):
        for w in (1.0, 0.2):
            pred, gt = make_batch(seed)
            ours = SAMLoss(ce_neg_weight=w)(pred, gt)[0]
            v3 = V3SAMLoss(ce_neg_weight=w)(pred, gt)
            assert torch.allclose(ours, v3, atol=1e-6), \
                f"seed={seed} ce_neg_weight={w}: {ours.item():.8f} != {v3.item():.8f}"


def check_dice_is_row_wise_like_v3():
    """Pins the (M, H, W) shape into monai. A per-image Dice is a different objective, and
    reaching it by accident is exactly how the E13 branch lost 12.8 points."""
    pred, gt = make_batch(0)
    pos = gt.flatten(2, 3).sum(-1) > 0
    out = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='none')(
        pred[pos].squeeze(1), gt[pos].float())
    M, H = int(pos.sum()), gt.shape[-2]
    assert tuple(out.shape) == (M, H, 1), \
        f"expected one Dice per (sample, row), got {tuple(out.shape)}"


def check_seg_loss_stays_within_dice_range():
    """A Dice loss is bounded in [0, 1]. The E13 branch reported up to 2.0 because it summed a
    second, undeclared Dice term over the negatives - the cheapest possible smoke alarm."""
    for seed in range(8):
        pred, gt = make_batch(seed)
        _, parts = SAMLoss()(pred, gt)
        assert 0.0 <= parts["seg_loss"] <= 1.0, f"seed={seed}: {parts}"


def check_no_dice_on_negatives_by_default():
    """v3 had none, and the term is a near-constant +1.0 with no gradient."""
    assert SAMLoss().dice_neg_weight == 0.0
    pred, gt = make_batch(0, all_negative=True)
    total, parts = SAMLoss()(pred, gt)
    assert parts["seg_loss"] == 0.0, parts
    assert parts["ce_loss_pos"] == 0.0, parts      # nothing positive to score
    assert torch.isfinite(total)                   # and no NaN from the empty branch


def check_weights_scale_rather_than_gate():
    """`dice_pos_weight=0.5` used to mean 1.0, because the weight was only read as an on/off."""
    pred, gt = make_batch(0)
    full = SAMLoss(dice_pos_weight=1.0)(pred, gt)[1]["seg_loss"]
    half = SAMLoss(dice_pos_weight=0.5)(pred, gt)[1]["seg_loss"]
    assert abs(half - full * 0.5) < 1e-6 * max(1.0, full), f"{half} vs {full}"


def check_ignored_pixels_do_not_contribute():
    """An ignored region must not move the loss, whatever the model predicts under it."""
    pred, gt = make_batch(0, with_ignore=True)
    loss_fn = SAMLoss()
    a = loss_fn(pred, gt)[0]
    pred2 = pred.clone()
    pred2[:, :, :, -4:, :] = 50.0                  # confidently wrong, entirely inside the ignore
    b = loss_fn(pred2, gt)[0]
    assert torch.allclose(a, b, atol=1e-6), f"{a.item()} != {b.item()}"


def check_all_negative_batch_is_finite():
    """The ragged tail of an 18-group epoch is often all-negative; it used to yield NaN."""
    pred, gt = make_batch(0, all_negative=True)
    total, parts = SAMLoss()(pred, gt)
    assert torch.isfinite(total)
    assert all(v == v for v in parts.values()), parts


def main():
    checks = [v for k, v in sorted(globals().items()) if k.startswith("check_")]
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}\n        {e}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
