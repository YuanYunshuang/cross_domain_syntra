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
#   1  inference: re-score curve 1000t x3, S0 x3 and S3 x3 without the test tiles that overlap
#      training (>10%: 513 tiles, >50%: 293). Each checkpoint is scored as it was trained: S0 and
#      S3 capped, the curve not. Filtered sets from make_overlap_filtered_tests.py.
#   2  SMOL on the full 14,675-tile split, no cap, r = 0.5            3 x 7.2 h
#   3  SMOL on the balanced 300 / 1,000 / 3,000 selections, no cap   9 x 7.2 h
#   4  SMOL no cap at ~5,100 tiles: balanced (5,102) and per-sheet   6 x 7.2 h
#   5  ours per-sheet 5,087, no cap (anchor recipe), seeds 1235/1236  2 x 12.6 h
#   6  ours balanced 300 and 3,000, no cap, seeds 1235/1236           4 x 12.6 h
#
# ~230 h. Every run skips itself once its result file exists, so the script can be restarted.
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
suffix () { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }   # the default seed carries no suffix
seedarg () { [ "$1" = 1234 ] && echo "" || echo "--seed $1"; }

# train_ours <task> <sel> <seed> [extra args]  - k=3 retrieved, then scored on its own selection
train_ours () {
  local t=$1 sel=$2 seed=$3; shift 3
  [ -f "logs/$t/$RES" ] && { echo "$t done, skip"; return; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$OURS" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" --num-src 3 --src-selection retrieved \
    $(seedarg "$seed") "$@" > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$OURS" -w "logs/$t/model_latest.pth" \
    -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" --num-src 3 --src-selection retrieved \
    "$@" > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(miou "logs/$t/$RES")"
}

# train_smol <task> <train data-info> <source list> <test data-info> <seed>  - no cap, r = 0.5
train_smol () {
  local t=$1 info=$2 src=$3 test_info=$4 seed=$5
  [ -f "logs/$t/$RES" ] && { echo "$t done, skip"; return; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SMOL" -t "$t" -d "$D" \
    --data-info "$info" --pos-prob 0.5 --source-file "$src" --target-file "$src" \
    --src-fg-max inf $(seedarg "$seed") > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" -w "logs/$t/model_latest.pth" \
    -d "$D" --data-info "$test_info" --source-file "$src" --src-fg-max inf > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(miou "logs/$t/$RES")"
}

# ---- 1. overlap-filtered re-scoring (inference only)
for ov in 10 50; do
  for run in curve_nocap_1000t curve_nocap_1000t_s1235 curve_nocap_1000t_s1236; do
    out="results/overlap/${run}_ov$ov"; [ -f "$out/$RES" ] && continue
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$OURS" -w "logs/$run/model_latest.pth" \
      -o "$out" -d "$D" --data-info "sample_pair_budget_random_gap0_1000t_s2026_test_samples_ov${ov}_gap0" \
      --num-src 3 --src-selection retrieved > /dev/null 2>&1
    echo "$(date '+%F %T') $out: $(miou "$out/$RES")"
  done
  for run in S3_k3_retrieved S3_k3_retrieved_s1235 S3_k3_retrieved_s1236; do
    out="results/overlap/${run}_ov$ov"; [ -f "$out/$RES" ] && continue
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$OURS" -w "logs/$run/model_latest.pth" \
      -o "$out" -d "$D" --data-info "sample_pair_cls_selection_100_test_samples_ov${ov}_gap0" \
      --num-src 3 --src-selection retrieved --src-fg-max 0.9 > /dev/null 2>&1
    echo "$(date '+%F %T') $out: $(miou "$out/$RES")"
  done
  for run in S0_smol_k1 S0_smol_k1_s1235 ratio_smol_p0.5; do
    out="results/overlap/${run}_ov$ov"; [ -f "$out/$RES" ] && continue
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" -w "logs/$run/model_latest.pth" \
      -o "$out" -d "$D" --data-info "sample_pair_cls_selection_100_test_samples_ov${ov}_gap0" \
      --source-file cls_selection_100.json --target-file "test_samples_ov${ov}.txt" > /dev/null 2>&1
    echo "$(date '+%F %T') $out: $(miou "$out/$RES")"
  done
done
echo "$(date '+%F %T') 1 done: overlap re-scoring"

# ---- 2. SMOL, full split
for s in "${SEEDS[@]}"; do
  train_smol "smol_full_nocap$(suffix $s)" sample_pair_train_samples_train_samples_gap0 \
    train_samples.txt sample_pair_train_samples_test_samples_gap0 "$s"
done
echo "$(date '+%F %T') 2 done: SMOL full split"

# ---- 3. SMOL, balanced curve
for b in 300 1000 3000; do
  sel="budget_random_gap0_${b}t_s2026"
  for s in "${SEEDS[@]}"; do
    train_smol "smol_curve_nocap_${b}t$(suffix $s)" "sample_pair_${sel}_${sel}_gap0" "${sel}.json" \
      "sample_pair_${sel}_test_samples_gap0" "$s"
  done
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
  [ -f "$d/$RES" ] && mkdir -p "results/revision/$(basename $d)" && cp "$d/$RES" "results/revision/$(basename $d)/"
done
echo "$(date '+%F %T') revision queue finished"
