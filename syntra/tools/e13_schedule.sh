#!/usr/bin/env bash
# E13 - training-sample selection under a fixed annotation budget.
#
# One unit of budget is one (tile, class) pair: one class digitised on one tile. See section
# 3d/3e of docs/EXPERIMENT_PLAN.md for why, and for how the iteration count is calibrated.
#
#   bash syntra/tools/e13_schedule.sh prepare   # selections + pairings, no GPU, ~10 min
#   bash syntra/tools/e13_schedule.sh pilot     # phase 1, 2 runs, ~25 h
#   bash syntra/tools/e13_schedule.sh ladder    # phase 3, 7 runs, ~70 h
#
# `prepare` is idempotent and writes into POOL_ROOT, never into the dataset itself. `pilot` and
# `ladder` run one job at a time on GPU 0; set CUDA_VISIBLE_DEVICES to place them elsewhere, and
# run the two halves of `ladder` on two machines if a second GPU is available.
set -euo pipefail

DATA=/home/yuan/data/syntra
INDEX=$DATA/global_index.npz
POOL_ROOT=/home/yuan/data/syntra_e13          # selections and pairings live here, not in $DATA
CFG=syntra/syntra_maps_s100                   # base config; everything else is overridden below
CELL=sheet                                    # the cell: which sheets share a retrieval pool.
                                              # One choice - it sets both the balance group and
                                              # the retrieval scope, and they must not disagree.
KW="hismap,siegfried"
XKW=hameln.3921_2023
SEL_PY=/home/yuan/anaconda3/envs/hismap/bin/python    # data tools: needs sklearn/matplotlib
TRAIN_PY=/home/yuan/anaconda3/envs/syntra/bin/python  # training: needs torch
export PYTHONPATH=.

# name | strategy | cell | guard | budget
# Experiment 1 first: the same number of training pairs as the n-shot scheme, chosen differently.
RUNS=(
  "e13x1_fps_b6154   fps      sheet   0  6154"
  "e13x1_unif_b6154  uniform  sheet   0  6154"
)
# Experiments 2-4 follow, anchored on whichever arm of experiment 1 wins; see docs/E13_PLAN.md.
PILOTS=("e13x1_fps_b6154")                    # calibration uses the run it will be applied to

prepare() {
  for spec in "${RUNS[@]}"; do
    read -r name strat cell guard budget <<< "$spec"
    echo "=== $name: $strat / cell $cell / guard $guard / B=$budget ==="
    $SEL_PY data_tools/indexing/select_budget.py -d "$DATA" --index "$INDEX" \
      --budget "$budget" --strategy "$strat" --density-guard "$guard" \
      --cell "$cell" --min-per-cell-class 1 --val-fraction 0.1 \
      --name "$name" --output-root "$POOL_ROOT" -dkw "$KW" -xkw "$XKW"
    # Sources are the run's own annotations: what you did not pay to label you cannot prompt
    # with either. Scope stays subset-local, so selection is the only variable across runs.
    for tgt in "$name" "${name}_val"; do
      $SEL_PY data_tools/indexing/make_pool_pairing.py -d "$POOL_ROOT" --index "$INDEX" \
        --source "$name" --target "$tgt" --scope "$cell" -k 8 -dkw "$KW" --suffix "$cell"
    done
    # test targets come from the dataset's own test lists, so they must be visible to the tool
    $SEL_PY data_tools/indexing/make_pool_pairing.py -d "$DATA" --index "$INDEX" \
      --source "$name" --target test_samples --scope "$cell" -k 8 -dkw "$KW" -xkw "$XKW" \
      --output-root "$POOL_ROOT" --suffix "$cell" || true
  done
}

# train <task> <data_info> <val_info> <iterations> <scheduler>
train() {
  local task=$1 info=$2 val=$3 itr=$4 sched=$5
  echo "=== train $task: $itr iterations, lr $sched ==="
  $TRAIN_PY syntra/train_one_gpu.py --config "$CFG" -t "$task" -d "$POOL_ROOT" \
    --data-info "$info" --validation-data "$val" --val-interval 2000 \
    --max-iterations "$itr" --lr-scheduler "$sched" --use-amp true -dkw "$KW" -xkw "$XKW"
}

# test <task> <data_info>
run_test() {
  local task=$1 info=$2
  echo "=== test $task ==="
  $TRAIN_PY syntra/test.py --config "$CFG" -d "$POOL_ROOT" --data-info "$info" \
    -w "logs/$task/model_latest.pth" -dkw "$KW" -xkw "$XKW" -s
}

pilot() {
  # Phase 1: constant LR after warmup, so the validation curve measures convergence and not the
  # decay. Long enough to see the knee; the fitted rule then sets the real lengths.
  for name in "${PILOTS[@]}"; do
    local itr=100000
    itr=150000
    train "${name}_pilot" "sample_pair_${name}_${name}_${CELL}" \
          "sample_pair_${name}_${name}_val_${CELL}" "$itr" constant
  done
  echo
  echo "Now read logs/<task>/val_curve.txt: take the smallest iteration count within 0.5 mIoU"
  echo "of each curve's peak, fit iterations(B) through the two points, and put it in ITERS."
}

# Phase 3: set from the pilots before running `ladder`. 100000 is the pre-calibration default
# and is deliberately left obviously provisional.
declare -A ITERS=( [e13x1_fps_b6154]=100000 [e13x1_unif_b6154]=100000 )

ladder() {
  for spec in "${RUNS[@]}"; do
    read -r name _ cell _ _ <<< "$spec"
    train "$name" "sample_pair_${name}_${name}_${CELL}" \
          "sample_pair_${name}_${name}_val_${CELL}" "${ITERS[$name]}" exp
    run_test "$name" "sample_pair_${name}_test_samples_${CELL}"
  done
  $SEL_PY syntra/tools/collect_results.py logs -o results/e13_runs.csv || true
}

case "${1:-}" in
  prepare) prepare ;;
  pilot)   pilot ;;
  ladder)  ladder ;;
  *) echo "usage: $0 {prepare|pilot|ladder}" >&2; exit 2 ;;
esac
