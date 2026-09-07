#!/bin/bash
# SMOL on the random-arm selections, after E-B finishes.
#
# The baseline for E4, at matched budget and matched training settings. Mirrors the four random
# arms of E-B exactly - same selections, same pairings, same corpus - so SYNTRA and SMOL differ
# only in what the method is: dataset_name=smol and num_src=1 (one prompt, no retrieval).
#
# Why re-run rather than cite /koko/SYNTRA_logs/smol: those runs used foreground_ratio as a dict
# with siegfried=0.01, which section 8 records as resolving to two different thresholds inside a
# single run, and 406 railway and 19 vineyard test tiles flip class between 0.01 and 0.05. The
# archived smol_hismap_n1_s100 file also holds two different runs - one `uniform` sampler on
# hismap without Resize scoring 85.09 over 11,342 samples, one `group_fixed_m` on hismap,siegfried
# with Resize scoring 81.80 over 37,935. Neither is comparable to a scalar-threshold SYNTRA run.
#
# --source-file / --target-file are mandatory here. SMOL reads sample lists rather than the
# pairing JSON and cannot infer a budgeted selection from the pairing's filename; before the fix
# it silently fell back to the whole 14,675-tile split.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
CFG=syntra/smol_budget
D=/home/yuan/data/syntra

ARMS=(
  "budget_random_gap0_300t_s2026"
  "budget_random_gap0_1000t_s2026"
  "budget_random_gap0_300t_s2027"
  "budget_random_gap0_300t_s2028"
)

# Wait for E-B: both the queue script and any training it still has on the card.
while pgrep -f "run_eb_queue.sh" > /dev/null || pgrep -f "train_one_gpu.py" > /dev/null; do
  sleep 300
done
echo "$(date '+%F %T') E-B finished, starting SMOL"

for sel in "${ARMS[@]}"; do
  task="smol_${sel#budget_}"
  if [ -f "logs/$task/test_model_latest_hismap,siegfried.txt" ]; then
    echo "$(date '+%F %T') $task already scored, skipping"
    continue
  fi
  echo "$(date '+%F %T') === train $task"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py \
    --config "$CFG" -t "$task" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" \
    --source-file "${sel}.json" --target-file "${sel}.json" \
    > "logs/$task.out" 2>&1
  echo "$(date '+%F %T') === test $task"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py \
    --config "$CFG" -w "logs/$task/model_latest.pth" -d "$D" \
    --data-info "sample_pair_${sel}_test_samples_gap0" \
    --source-file "${sel}.json" \
    > "logs/${task}_test.out" 2>&1
  echo "$(date '+%F %T') === done $task: $(grep -A1 'Overal Mean' "logs/$task/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') SMOL queue finished"
