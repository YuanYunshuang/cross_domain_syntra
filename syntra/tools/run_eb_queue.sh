#!/bin/bash
# E-B on the 4090, one run at a time.
#
# The GPU holds one run at these settings - 15.4 GB of 24.5 - so this waits for the anchor to
# finish rather than competing with it. Each arm trains 100k and is then scored on model_latest,
# per P0.5: validation_data is null, so "best" would select on training loss.
#
# Ordering is by information, not by seed: the first four give both rungs of the comparison, and
# 5-8 are the replicates that put error bars on B=300. Killing this after any run leaves a
# coherent result set rather than a half-finished cell.
#
#   1-2  random vs fps+g15 at   300 tiles, seed 2026   <- the headline comparison
#   3-4  random vs fps+g10 at 1,000 tiles, seed 2026   <- does the effect survive a larger budget
#   5-8  the other two selection seeds at 300 tiles
#
# Every arm shares one test pairing (11,342 targets, 37,935 pairs), so the arms are scored on
# identical data - section 8, and the third of P0.3's asserts.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
CFG=syntra/syntra_anchor_cls100          # base config; the pairing is overridden per arm
D=/home/yuan/data/syntra

ARMS=(
  "budget_random_gap0_300t_s2026"
  "budget_fps_g15_gap0_300t_s2026"
  "budget_random_gap0_1000t_s2026"
  "budget_fps_g10_gap0_1000t_s2026"
  "budget_random_gap0_300t_s2027"
  "budget_fps_g15_gap0_300t_s2027"
  "budget_random_gap0_300t_s2028"
  "budget_fps_g15_gap0_300t_s2028"
)

# Wait for the GPU. Checking for our own training processes rather than free memory, because
# memory frees a moment before the process exits and a race would OOM the next run.
while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting E-B"

for sel in "${ARMS[@]}"; do
  task="eb_${sel#budget_}"
  if [ -f "logs/$task/test_model_latest_hismap,siegfried.txt" ]; then
    echo "$(date '+%F %T') $task already scored, skipping"
    continue
  fi
  echo "$(date '+%F %T') === train $task"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py \
    --config "$CFG" -t "$task" -d "$D" --data-info "sample_pair_${sel}_${sel}_gap0" \
    > "logs/$task.out" 2>&1
  echo "$(date '+%F %T') === test $task"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py \
    --config "$CFG" -w "logs/$task/model_latest.pth" -d "$D" \
    --data-info "sample_pair_${sel}_test_samples_gap0" \
    > "logs/${task}_test.out" 2>&1
  echo "$(date '+%F %T') === done $task: $(grep -A1 'Overal Mean' "logs/$task/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') E-B queue finished"
