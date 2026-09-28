#!/bin/bash
# The 4090's share of the revision run list (syntra-review, 2026-09-28), in priority order.
#
# Decisions behind it: BOTH models run without the source-coverage cap (ours is better uncapped;
# the existing capped pair stays as a sensitivity check), three training seeds everywhere (the
# default 1234, then 1235 and 1236), and every comparison on one box. The k=3 random-vs-retrieved
# study runs on the 5090 and is not here.
#
# SMOL no-cap is `--src-fg-max inf`, at train AND test. SMOL-MapSeg's 0.9 cap was a literal in
# smol.py until the commit that made it a parameter, so the archived SMOL runs are all capped.
#
#   1  (moved to the 5090) overlap-filtered re-scoring of curve 1000t, S0 and S3
#   2  SMOL on the full 14,675-tile split, no cap, r = 0.5            3 x 7.2 h
#   3  SMOL on the balanced 1,000 / 3,000 / 300 selections, no cap   9 x 7.2 h
#   4  SMOL no cap at ~5,100 tiles: balanced (5,102) and per-sheet   6 x 7.2 h
#   5  ours per-sheet 5,087, no cap (anchor recipe), seeds 1235/1236  2 x 12.6 h
#   6  ours balanced 300 and 3,000, no cap, seeds 1235/1236           4 x 12.6 h
#
# ~227 h. A run whose result holds its "Overal Mean" block is skipped, and a run whose training log
# says it stopped at 100k is only scored (see scored/trained below), so the queue shell can be
# stopped and restarted while a training job keeps running. A training job that is itself killed
# restarts from scratch, and a scoring job that dies is re-scored.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
OURS=syntra/syntra_anchor_cls100
SMOL=syntra/smol_budget
RES="test_model_latest_hismap,siegfried.txt"
SEEDS=(1234 1235 1236)

until [ -L logs ]; do sleep 120; done
while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') starting"

miou () { grep -A1 'Overal Mean' "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }
# A run counts as scored only once its result file holds the "Overal Mean" block. test.py opens the
# file and logs "# testing samples" before scoring starts, so mere existence also matches a run
# whose scoring died half-way, which a restart would then skip while it looks complete.
scored () { grep -q 'Overal Mean' "logs/$1/$RES" 2>/dev/null; }
# A run counts as trained only once train_one_gpu.py has logged that it stopped. That line comes
# after the final model_latest.pth and model_100000.pth are fully written, whereas the checkpoint
# file exists from the moment torch.save opens it and can be partial after a crash.
trained () { grep -q 'reached 100000 iterations, stopping' "logs/$1/log.txt" 2>/dev/null; }
suffix () { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }   # the default seed carries no suffix
seedarg () { [ "$1" = 1234 ] && echo "" || echo "--seed $1"; }

# train_ours <task> <sel> <seed> [extra args]  - k=3 retrieved, then scored on its own selection
train_ours () {
  local t=$1 sel=$2 seed=$3; shift 3
  scored "$t" && { echo "$t done, skip"; return; }
  if trained "$t"; then echo "$(date '+%F %T') === $t trained, scoring only"; else
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$OURS" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" --num-src 3 --src-selection retrieved \
    $(seedarg "$seed") "$@" > "logs/$t.out" 2>&1
  fi
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$OURS" -w "logs/$t/model_latest.pth" \
    -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" --num-src 3 --src-selection retrieved \
    "$@" > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(miou "logs/$t/$RES")"
}

# train_smol <task> <train data-info> <source list> <test data-info> <seed>  - no cap, r = 0.5
train_smol () {
  local t=$1 info=$2 src=$3 test_info=$4 seed=$5
  scored "$t" && { echo "$t done, skip"; return; }
  if trained "$t"; then echo "$(date '+%F %T') === $t trained, scoring only"; else
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SMOL" -t "$t" -d "$D" \
    --data-info "$info" --pos-prob 0.5 --source-file "$src" --target-file "$src" \
    --src-fg-max inf $(seedarg "$seed") > "logs/$t.out" 2>&1
  fi
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" -w "logs/$t/model_latest.pth" \
    -d "$D" --data-info "$test_info" --source-file "$src" --src-fg-max inf > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(miou "logs/$t/$RES")"
}

# ---- 1. overlap-filtered re-scoring: moved to the 5090 (2026-09-28). Evaluation is portable
# across the two boxes (a 4090 checkpoint re-scored on the 5090 matched to <= 0.02 per class), and
# the 5090 holds a copy of the 4090 logs. Filtered sets: data_tools/indexing/make_overlap_filtered_tests.py.

# ---- 2. SMOL, full split
for s in "${SEEDS[@]}"; do
  train_smol "smol_full_nocap$(suffix $s)" sample_pair_train_samples_train_samples_gap0 \
    train_samples.txt sample_pair_train_samples_test_samples_gap0 "$s"
done
echo "$(date '+%F %T') 2 done: SMOL full split"

# ---- 3. SMOL, balanced curve. 1,000 and 3,000 first (user's 3-day plan, 2026-09-28): together
# with block 2 they finish in ~67 h; 300 follows.
for b in 1000 3000 300; do
  sel="budget_random_gap0_${b}t_s2026"
  for s in "${SEEDS[@]}"; do
    train_smol "smol_curve_nocap_${b}t$(suffix $s)" "sample_pair_${sel}_${sel}_gap0" "${sel}.json" \
      "sample_pair_${sel}_test_samples_gap0" "$s"
  done
  echo "$(date '+%F %T') 3 block done: SMOL curve ${b}t"
done
echo "$(date '+%F %T') 3 done: SMOL curve"

# ---- 4. SMOL allocation at ~5,100 tiles
sel=budget_random_gap0_5087t_s2026
for s in "${SEEDS[@]}"; do
  train_smol "smol_curve_nocap_5087t$(suffix $s)" "sample_pair_${sel}_${sel}_gap0" "${sel}.json" \
    "sample_pair_${sel}_test_samples_gap0" "$s"
done
sel=cls_selection_100
for s in "${SEEDS[@]}"; do
  train_smol "S0_smol_nocap$(suffix $s)" "sample_pair_${sel}_${sel}_gap0" "${sel}.json" \
    "sample_pair_${sel}_test_samples_gap0" "$s"
done
echo "$(date '+%F %T') 4 done: SMOL allocation"

# ---- 5. ours, per-sheet, no cap. The default seed is syntra_anchor_cls100 (83.97 at k=3).
for s in 1235 1236; do
  train_ours "anchor_cls100_s$s" cls_selection_100 "$s"
done
echo "$(date '+%F %T') 5 done: ours per-sheet"

# ---- 6. ours, balanced curve, no cap. 1,000 already has three seeds.
for b in 300 3000; do
  for s in 1235 1236; do
    train_ours "curve_nocap_${b}t_s$s" "budget_random_gap0_${b}t_s2026" "$s"
  done
done
echo "$(date '+%F %T') 6 done: ours curve seeds"

# Result files for the paper session, which reads from results/.
mkdir -p results/revision
for d in logs/smol_full_nocap* logs/smol_curve_nocap_* logs/S0_smol_nocap* logs/anchor_cls100_s* \
         logs/curve_nocap_300t_s* logs/curve_nocap_3000t_s*; do
  grep -q 'Overal Mean' "$d/$RES" 2>/dev/null && mkdir -p "results/revision/$(basename $d)" && cp "$d/$RES" "results/revision/$(basename $d)/"
done
echo "$(date '+%F %T') revision queue finished"
