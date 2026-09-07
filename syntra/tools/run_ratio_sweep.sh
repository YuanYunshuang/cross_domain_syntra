#!/bin/bash
# Ablation step 0: find the positive:negative ratio, then pin it for every later rung.
#
# The ratio is currently an uncontrolled 30-point difference under the whole comparison. SMOL
# trains at 80% positive because select_tgt_cls draws a positive with probability pos_prob=0.8;
# the anchor trains at exactly 50% because it happens to have 9 positive and 9 negative sampler
# groups. Nobody chose either. Section 1b already measured this axis as worth +2.16 mIoU on its
# own, so it is a leading candidate for the entire SYNTRA-over-SMOL margin and has to be settled
# before the grouping, source-count and retrieval steps mean anything.
#
# Swept on SMOL, not on SYNTRA, and deliberately: tuning the *baseline* and still winning is the
# conservative direction. Tuning our own method and not the baseline is what a reviewer would
# call cherry-picking.
#
# The grid spans both sides of 50%. The prior is that positives should dominate - the task is to
# find positive pixels - but the evidence in hand points the other way: section 1b gained +2.16
# by moving the negative:positive ratio 0.97 -> 1.53, and the anchor's worst class fails by
# over-prediction, precision 71.62 against recall 90.87. Either outcome is worth reporting.
#
# 50k rather than 100k, on P1.2's argument that halving the schedule preserves the *ordering* of
# conditions even though absolute mIoU drops. That argument has never been checked. If it is
# wrong, r* is wrong and every later rung inherits it - so validate it by re-running the two
# endpoints at 100k before trusting the middle of the grid.
#
# Realised positive shares, measured over 20k draws rather than assumed, since select_tgt_cls
# forces a positive when the tile carries every class:  0.33 -> 36.0%, 0.5 -> 52.1%,
# 0.67 -> 68.1%, 0.8 -> 80.0%. The matching cd_syntra2 sampler settings are 1:2, 1:1, 2:1, 4:1.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
CFG=syntra/smol_budget
D=/home/yuan/data/syntra
SEL=cls_selection_100
ITR=50000

# Wait for whatever holds the card - E-B arm 1 is finishing and its queue is already stopped.
while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting the ratio sweep"

for pp in 0.33 0.5 0.67 0.8; do
  task="ratio_smol_p${pp}"
  if [ -f "logs/$task/test_model_latest_hismap,siegfried.txt" ]; then
    echo "$(date '+%F %T') $task already scored, skipping"
    continue
  fi
  echo "$(date '+%F %T') === train $task (pos_prob=$pp)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py \
    --config "$CFG" -t "$task" -d "$D" -itr "$ITR" --pos-prob "$pp" \
    --data-info "sample_pair_${SEL}_${SEL}_gap0" \
    --source-file "${SEL}.json" --target-file "${SEL}.json" \
    > "logs/$task.out" 2>&1
  echo "$(date '+%F %T') === test $task"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py \
    --config "$CFG" -w "logs/$task/model_latest.pth" -d "$D" \
    --data-info "sample_pair_${SEL}_test_samples_gap0" --source-file "${SEL}.json" \
    > "logs/${task}_test.out" 2>&1
  echo "$(date '+%F %T') === done $task: $(grep -A1 'Overal Mean' "logs/$task/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') ratio sweep finished"
