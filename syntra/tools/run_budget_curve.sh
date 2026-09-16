#!/bin/bash
# Budget curve at the S3 recipe: how little annotation does SYNTRA need to match SMOL?
#
# "+2.76 mIoU at equal budget" is a modest claim and, at n=3 with ~2 points of run-to-run spread,
# not a significant one. "Matches SMOL's full-data accuracy with N times less annotation" is a
# strong claim and the data suggests it is there: SMOL saturates by ~3,000 tiles - 82.54 at 3,186
# and 82.57 at 14,675 - while S3 reaches 84.92 at 5,087. The question is how far left S3 can go
# before it drops to SMOL's ceiling.
#
# Recipe held at S3 exactly: k=3, retrieved, src_fg_max 0.9, 50/50 positive share from the 9+9
# sampler groups, 100k, and the same 37,935-pair test set. Only the budget moves.
#
# Selections are the corpus-class balanced random ones generated for E-B, so the curve uses one
# selection scheme throughout:
#     300   budget_random_gap0_300t_s2026     880 training pairs
#   1,000   budget_random_gap0_1000t_s2026  2,958
#   3,000   budget_random_gap0_3000t_s2026  8,733
#
# CAVEAT for the write-up: the 5,087 point already in hand (S3) uses cls_selection_100, which is
# the n-shot scheme - a quota per (sheet, class) rather than per (corpus, class). So the top of the
# curve is a different allocation from the rest, and railway/vineyard get 10 tiles each there
# against ~1/9 of the budget here. If the curve is quoted as a single line, add a matched
# budget_random point at 5,087; if the top point stays n-shot, say so in the caption.
#
# The earlier 76.24 at 300 tiles is NOT on this curve: it predates the coverage cap and was scored
# at k=7. Not comparable, and it is the reason these are fresh runs rather than a re-plot.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
CFG=syntra/syntra_anchor_cls100
TEST_INFO=sample_pair_cls_selection_100_test_samples_gap0

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting the budget curve"

for b in 300 1000 3000; do
  sel="budget_random_gap0_${b}t_s2026"
  t="curve_S3_${b}t"
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$t done, skip"; continue; }
  echo "$(date '+%F %T') === train $t ($b tiles)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" \
    --num-src 3 --src-selection retrieved --src-fg-max 0.9 > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
    --num-src 3 --src-selection retrieved --src-fg-max 0.9 > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') budget curve finished"
