#!/bin/bash
# Two things: seed the curve's decisive point, and train the checkpoint E5 needs.
#
# 1-2. curve_nocap_1000t at two more seeds. The efficiency headline rests on 82.96 at 1,000 tiles
#      against SMOL-full's 82.57 on 14,675 - a 0.39 gap against ~2 points of run-to-run spread.
#      Three monotone points (76.90 / 82.96 / 85.78) make it safer than that gap suggests, since
#      the claim is a threshold crossing rather than a mean difference, but single-seed is the
#      first thing a reviewer will push on. No cap: section 6b-quater retracts it from the recipe.
#
# 3.   E5's held-out checkpoint. Section 6a budgets E5 at zero training, which is only right if
#      the checkpoint never saw the held-out sheet - otherwise "unseen" means unseen at retrieval
#      time only, which is a much weaker claim and a reviewer will spot it. One run fixes that;
#      the pool-size sweep over it (0/1/10/50/100 tiles x {retrieved, random}) is then genuinely
#      free, since retrieval is baked into the pairing offline and evaluation noise is < 0.01.
#
#      hameln.3922_1996 is held out: 420 train and 480 test tiles carrying all four Hameln classes
#      (forest 332, grass 195, settlement 178, water 85), so the sweep can say something per class
#      rather than about one dominant one. Sixteen Hameln sheets remain in training, so no class
#      loses coverage.
#
# Everything else matches the curve recipe exactly - k=3, retrieved, r* = 0.5, 100k, prompts from
# the run's own budget at test time.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
CFG=syntra/syntra_anchor_cls100
SEL1K=budget_random_gap0_1000t_s2026
SEL=cls_selection_100
HOLDOUT=hameln.3922_1996

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting"

for seed in 1235 1236; do
  t="curve_nocap_1000t_s${seed}"
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$t done, skip"; continue; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${SEL1K}_${SEL1K}_gap0" \
    --num-src 3 --src-selection retrieved --seed "$seed" > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" \
    --data-info "sample_pair_${SEL1K}_test_samples_gap0" \
    --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done

t="e5_holdout_${HOLDOUT}"
if [ ! -f "logs/$t/test_model_latest_hismap,siegfried.txt" ]; then
  echo "$(date '+%F %T') === train $t (holding out $HOLDOUT)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${SEL}_${SEL}_gap0" -xkw "$HOLDOUT" \
    --num-src 3 --src-selection retrieved > "logs/$t.out" 2>&1
  # Reference score on the standard test set. The held-out sheet is still in it, so this run's
  # per-class numbers already contrast seen against unseen; the E5 sweep proper comes after.
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${SEL}_test_samples_gap0" \
    --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
fi
echo "$(date '+%F %T') finished"
