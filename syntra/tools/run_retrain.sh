#!/bin/bash
# Retrain one of our (cd_syntra2, k=3 retrieved, no cap) runs from scratch, with a collapse guard.
#
#   bash syntra/tools/run_retrain.sh <task> <selection> <seed>
#   e.g. bash syntra/tools/run_retrain.sh curve_nocap_3000t_s1236 budget_random_gap0_3000t_s2026 1236
#
# Why the guard exists: curve_nocap_3000t_s1236 (2026-10-07) trained to 100k without a NaN, passed
# run_revision.sh's `trained` check and was scored - at 0.00 mIoU. Between 1k and 2k iterations,
# at the end of warm-up, it fell into predicting no foreground and never left. Completing training
# says nothing about whether the model learned anything, so a scored run is also checked here:
# mIoU below COLLAPSE_MIOU, or every class at IoU 0, marks it DIVERGED. A diverged attempt is moved
# to <task>.diverged-<timestamp> (logs/ and results/revision/), kept as evidence, and retrained from
# scratch - never resumed, since --resume restores no RNG state - up to MAX_ATTEMPTS in total.
#
# MAX_ATTEMPTS defaults to 3. Retraining only on failure is a selection rule, and the paper has to
# state how many attempts any reported seed took; every attempt is logged here and in logs/.
set -u
cd "${REPO:-/home/yuan/code/CD_SYNTRA}"
PY=${PY:-/home/yuan/anaconda3/envs/syntra/bin/python}
D=${D:-/home/yuan/data/syntra}
OURS=syntra/syntra_anchor_cls100
RES="test_model_latest_hismap,siegfried.txt"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
COLLAPSE_MIOU=${COLLAPSE_MIOU:-10.0}
t=$1 sel=$2 seed=$3

miou () { grep -A1 'Overal Mean' "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }
trained () { grep -q 'reached 100000 iterations, stopping' "logs/$1/log.txt" 2>/dev/null; }
# collapsed <result file>: mIoU below the threshold, or no class with a non-zero IoU
collapsed () {
  local m; m=$(miou "$1")
  [ -z "$m" ] && return 0
  awk -v m="$m" -v c="$COLLAPSE_MIOU" 'BEGIN{exit !(m < c)}' && return 0
  ! grep -A1 'Notion set' "$1" | grep 'IoU (FG)' | awk '{print $NF}' | grep -qv '^0\.00$'
}
set_aside () {   # set_aside <task> <tag>
  local stamp; stamp=$(date '+%Y%m%d-%H%M%S')
  # the attempt number is in <tag>, so two attempts set aside within one second cannot collide (an
  # existing target would make mv nest the folder inside it rather than rename it)
  for p in "logs/$1" "logs/$1.out" "logs/$1_test.out" "results/revision/$1"; do
    [ -e "$p" ] || continue
    case "$p" in
      *.out) mv "$p" "${p%.out}.$2-$stamp.out" ;;
      *)     mv "$p" "$p.$2-$stamp" ;;
    esac
  done
}
seedarg () { [ "$1" = 1234 ] && echo "" || echo "--seed $1"; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  [ -e "logs/$t" ] && { set_aside "$t" "failed-a$attempt"; echo "$(date '+%F %T') === $t: unfinished folder set aside"; }
  echo "$(date '+%F %T') === train $t attempt $attempt/$MAX_ATTEMPTS"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$OURS" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" --num-src 3 --src-selection retrieved \
    $(seedarg "$seed") > "logs/$t.out" 2>&1
  if ! trained "$t"; then
    echo "$(date '+%F %T') === $t attempt $attempt FAILED: training did not reach 100k"; continue
  fi
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$OURS" -w "logs/$t/model_latest.pth" \
    -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" --num-src 3 --src-selection retrieved \
    > "logs/${t}_test.out" 2>&1
  if collapsed "logs/$t/$RES"; then
    echo "$(date '+%F %T') === $t attempt $attempt DIVERGED: mIoU $(miou "logs/$t/$RES"); set aside"
    set_aside "$t" "diverged-a$attempt"
    continue
  fi
  mkdir -p "results/revision/$t" && cp "logs/$t/$RES" "results/revision/$t/"
  echo "$(date '+%F %T') === $t: $(miou "logs/$t/$RES") (attempt $attempt)"
  exit 0
done
echo "$(date '+%F %T') === $t: no healthy run in $MAX_ATTEMPTS attempt(s)"
exit 1
