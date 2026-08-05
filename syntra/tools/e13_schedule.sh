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
KW="hismap,siegfried"
XKW=hameln.3921_2023
FG=hameln=0.05,bayern=0.05,siegfried=0.01
SEL_PY=/home/yuan/anaconda3/envs/hismap/bin/python    # data tools: needs sklearn/matplotlib
TRAIN_PY=/home/yuan/anaconda3/envs/syntra/bin/python  # training: needs torch
export PYTHONPATH=.

# Experiment 1: the same number of positive training samples as the n-shot scheme, an equal
# number of negatives, and a different way of choosing them. Gap 0 is sheet-wise, the validated
# setting, so the selection is the only thing that differs.
#
# The n-shot arm is that scheme written out as the passes it consumes - 4,859 tiles, every class
# of their sheet - with its negatives sampled down to match its positives: 6,801 + 6,697. The
# proposed arm buys the same 13,498 passes at the same split. The original's 61.5% negative skew
# was an accident of exhaustive labelling, not a design choice.
NPOS=6801        # the n-shot scheme's positive count; the proposed arm is given the same
NNEG=6697        # one negative draw, shared verbatim by both arms
NSHOT=e13x1_nshot
# name | strategy | cell-gap | guard | budget   (positives only; negatives are merged in)
RUNS=("e13x1_fps  fps  0  0  $NPOS")
# Experiments 2-4 follow, anchored on whichever arm of experiment 1 wins; see docs/E13_PLAN.md.
PILOTS=("e13x1_fps")                    # calibration uses the run it will be applied to

prepare() {
  # One negative draw, at random, shared verbatim by both arms so the positives are the only
  # thing that differs between them.
  $SEL_PY data_tools/indexing/select_budget.py -d "$DATA" --index "$INDEX" \
    --budget "$NNEG" --strategy random --negative-fraction 1.0 --negative-strategy random \
    --cell-gap 0 --min-per-cell-class 0 --val-fraction 0.1 --fg-ratio "$FG" \
    --name e13x1_neg --output-root "$POOL_ROOT" -dkw "$KW" -xkw "$XKW"

  # the n-shot arm's positive half: its tiles times the classes they carry
  $SEL_PY data_tools/indexing/expand_selection.py -d "$DATA" --source cls_selection_100 \
    --name "${NSHOT}_pos" --output-root "$POOL_ROOT" --fg-ratio "$FG" --negative-ratio 0 \
    --val-fraction 0.1 -dkw "$KW" -xkw "$XKW"
  for part in "" "_val"; do
    $SEL_PY data_tools/indexing/merge_selections.py -d "$POOL_ROOT" \
      --inputs "${NSHOT}_pos${part},e13x1_neg${part}" --name "${NSHOT}${part}" \
      -dkw "$KW" -xkw "$XKW"
  done
  for tgt in "$NSHOT" "${NSHOT}_val" "test_samples"; do
    $SEL_PY data_tools/indexing/make_pool_pairing.py -d "$POOL_ROOT" --index "$INDEX" \
      --source "$NSHOT" --target "$tgt" --cell-gap 0 -k 8 -dkw "$KW" -xkw "$XKW" --suffix gap0
  done

  for spec in "${RUNS[@]}"; do
    read -r name strat gap guard budget <<< "$spec"
    echo "=== $name: $strat / cell-gap $gap / guard $guard / B=$budget ==="
    $SEL_PY data_tools/indexing/select_budget.py -d "$DATA" --index "$INDEX" \
      --budget "$budget" --strategy "$strat" --density-guard "$guard" \
      --cell-gap "$gap" --min-per-cell-class 1 --val-fraction 0.1 --fg-ratio "$FG" \
      --name "${name}_pos" --output-root "$POOL_ROOT" -dkw "$KW" -xkw "$XKW"
    for part in "" "_val"; do
      $SEL_PY data_tools/indexing/merge_selections.py -d "$POOL_ROOT" \
        --inputs "${name}_pos${part},e13x1_neg${part}" --name "${name}${part}" \
        -dkw "$KW" -xkw "$XKW"
    done
    # Sources are the run's own annotations: what you did not pay to label you cannot prompt
    # with either. The pairing gap matches the selection gap - they are one decision.
    for tgt in "$name" "${name}_val"; do
      $SEL_PY data_tools/indexing/make_pool_pairing.py -d "$POOL_ROOT" --index "$INDEX" \
        --source "$name" --target "$tgt" --cell-gap "$gap" -k 8 -dkw "$KW" --suffix "gap$gap"
    done
    # Test targets and the selection both have to be visible at once. mirror_subset symlinks
    # the dataset's own lists into POOL_ROOT, so everything is read from there; pointing this at
    # DATA finds test_samples.txt but not the selection, and writes an empty pairing in silence.
    $SEL_PY data_tools/indexing/make_pool_pairing.py -d "$POOL_ROOT" --index "$INDEX" \
      --source "$name" --target test_samples --cell-gap "$gap" -k 8 -dkw "$KW" -xkw "$XKW" \
      --suffix "gap$gap"
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
    local itr=150000
    train "${name}_pilot" "sample_pair_${name}_${name}_gap0" \
          "sample_pair_${name}_${name}_val_gap0" "$itr" constant
  done
  echo
  echo "Now read logs/<task>/val_curve.txt: take the smallest iteration count within 0.5 mIoU"
  echo "of each curve's peak, fit iterations(B) through the two points, and put it in ITERS."
}

# Phase 3: set from the pilots before running `ladder`. 100000 is the pre-calibration default
# and is deliberately left obviously provisional.
# From the calibration pilot: held-out mIoU was still rising at 150,000 with no plateau and no
# overfitting signal - training loss reached 0.073 while validation kept improving - so this is
# a compute-bound choice, not a converged one, and the paper has to say so.
declare -A ITERS=( [e13x1_fps]=150000 [e13x1_nshot]=150000 )

# Both arms of experiment 1. The n-shot arm is the scheme as it exists, the fps arm is the same
# number of positives chosen differently, and they share their negatives verbatim.
ARMS=("e13x1_nshot" "e13x1_fps")

ladder() {
  for name in "${ARMS[@]}"; do
    train "$name" "sample_pair_${name}_${name}_gap0" \
          "sample_pair_${name}_${name}_val_gap0" "${ITERS[$name]}" exp
    run_test "$name" "sample_pair_${name}_test_samples_gap0"
  done
  $SEL_PY syntra/tools/collect_results.py logs -o results/e13_runs.csv || true
}

case "${1:-}" in
  prepare) prepare ;;
  pilot)   pilot ;;
  ladder)  ladder ;;
  *) echo "usage: $0 {prepare|pilot|ladder}" >&2; exit 2 ;;
esac
