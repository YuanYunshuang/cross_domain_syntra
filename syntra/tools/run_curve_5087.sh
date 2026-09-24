#!/bin/bash
# The curve's top point, on the same selection scheme as the rest of it.
#
# Figure 1 currently shows 3,000 tiles (85.78) beating 5,087 (83.97), which is not a budget effect:
# the 5,087 point is cls_selection_100, the n-shot scheme, which quotas per (sheet, class) and so
# gives railway and vineyard 10 tiles each where corpus-class balancing gives them about a ninth of
# the budget. Every other point on the curve is corpus-class random. This run supplies the matched
# point so the curve is one scheme throughout, and the n-shot point becomes a separate comparison
# of allocation at equal budget rather than an anomaly in the middle of a figure.
#
# The selection lands on 5,102 tiles rather than 5,087: the bisection takes the smallest pair
# budget holding at least the target, and 5,850 pairs gives 5,102. A 0.3% overshoot, recorded here
# rather than chased.
#
# Recipe identical to the rest of the curve: k=3, retrieved, no cap (6b-quater retracts it),
# r* = 0.5 from the 9+9 sampler groups, 100k, scored with prompts from its own budget.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
CFG=syntra/syntra_anchor_cls100
sel=budget_random_gap0_5087t_s2026
t=curve_nocap_5087t

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting $t"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
  --data-info "sample_pair_${sel}_${sel}_gap0" \
  --num-src 3 --src-selection retrieved > "logs/$t.out" 2>&1
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
  -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" \
  --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
