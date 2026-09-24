#!/bin/bash
# The outline's gaps list (PAPER_OUTLINE.md, "Gaps worth one run each"), in cost order.
#
# Gap 1, the matched budget_random point at 5,087, is already done (curve_nocap_5087t, 86.09).
#
# Gap 4 - E7b, the temporal-gap sweep. Inference only. The outline proposed a 2017 target, but
#   E5's checkpoint trained on 2017, so that target is not unseen. Instead the held-out 1996
#   survey is the target throughout and only the *year the prompt pool comes from* varies:
#   1898 / 1974 / 1982 / 2005 / 2017 (gaps 98 / 22 / 14 / 9 / 21 years) against 1996's own pool.
#   Pools are nested and class-balanced from each year's training tiles
#   (data_tools/indexing/make_eval_pools.py), 10 and 100 tiles. The other years' tiles were seen
#   in training - that is the realistic setting, an archive with some years annotated - so this
#   asks whether an annotated neighbouring survey substitutes for annotating the new one.
#
# Gap 3 - held-out footprint. hameln.3821 with all four of its years (1974 / 2002 / 2012 / 2023)
#   excluded from training, so neither the ground nor any of its surveys is seen. Then the E5
#   sweep per year at 10 / 50 / 100 tiles, against the anchor (trained on all of it) as ceiling.
#   Paired with E5 this separates spatial from temporal generalisation. One training run.
#
# Gap 2 - matched SMOL at 300 / 1,000 / 3,000 tiles on the same selections as the curve, so
#   Figure 1 compares two curves. smol_budget with --pos-prob 0.5, exactly as S0; the old
#   run_smol_queue.sh predates the ratio sweep and would have run at SMOL's 0.8. Three runs.
#
# ~4 training runs at ~12.5 h each, one at a time on the 4090.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
CFG=syntra/syntra_anchor_cls100
SMOL=syntra/smol_budget
SEL=cls_selection_100
ANCHOR=logs/syntra_anchor_cls100/model_latest.pth
E5=logs/e5_holdout_hameln.3922_1996/model_latest.pth
FP=hameln.3821

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') starting"

score () {  # score <checkpoint> <out_dir> <subset> <data_info> <selection>
  [ -f "$2/test_model_latest_$3.txt" ] && return
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" -w "$1" -o "$2" -d "$D" \
    -dkw "$3" --data-info "$4" --num-src 3 --src-selection "$5" > /dev/null 2>&1
  echo "$(date '+%F %T') $2: $(grep -A1 'Overal Mean' "$2/test_model_latest_$3.txt" 2>/dev/null | tail -1)"
}

# ---- gap 4: E7b
# The loader only registers subsets that hold the pairing file, so a source tile from another
# year raises KeyError on its folder. Each pool year therefore holds an empty `{}` under the same
# name and is added to -dkw: its folder, labels and colours are registered, and it contributes no
# targets - still 1,920 target-class pairs, all 1996.
for y in 1898 1974 1982 1996 2005 2017; do
  kw=hameln.3922_1996; [ "$y" != 1996 ] && kw="$kw,hameln.3922_$y"
  for n in 10 100; do
    info="sample_pair_e7b_pool_${y}_${n}_e7b_targets_gap0"
    for s in retrieved random; do
      score "$E5" "results/e7b/e5_holdout_pool${y}_n${n}_${s}" "$kw" "$info" "$s"
    done
    score "$ANCHOR" "results/e7b/anchor_pool${y}_n${n}_retrieved" "$kw" "$info" retrieved
  done
done
echo "$(date '+%F %T') E7b done"

# logs/ becomes a symlink to /koko/cd_syntra_logs_09_2026 (the local disk ran out). E7b above only
# reads checkpoints and writes to results/, so it runs during the move. Training writes into
# logs/, so it must not start in a half-moved directory.
until [ -L logs ]; do sleep 120; done

# ---- gap 3: held-out footprint
t="fp_holdout_${FP}"
if [ ! -f "logs/$t/model_latest.pth" ]; then
  echo "$(date '+%F %T') === train $t (holding out every year of $FP)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${SEL}_${SEL}_gap0" -xkw "$FP" \
    --num-src 3 --src-selection retrieved > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${SEL}_test_samples_gap0" \
    --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
fi
mkdir -p "results/fp/$t" && cp "logs/$t/test_model_latest_hismap,siegfried.txt" "results/fp/$t/"
for y in 1974 2002 2012 2023; do
  for n in 10 50 100; do
    info="sample_pair_fp_pool_${n}_test_samples_gap0"
    score "logs/$t/model_latest.pth" "results/fp/fp_holdout_${y}_n${n}" "${FP}_$y" "$info" retrieved
    score "$ANCHOR" "results/fp/anchor_${y}_n${n}" "${FP}_$y" "$info" retrieved
  done
done
echo "$(date '+%F %T') held-out footprint done"

# ---- gap 2: matched SMOL curve
for b in 300 1000 3000; do
  sel="budget_random_gap0_${b}t_s2026"
  t="smol_curve_${b}t"
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$t done, skip"; continue; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SMOL" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" --pos-prob 0.5 \
    --source-file "${sel}.json" --target-file "${sel}.json" > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" \
    --source-file "${sel}.json" > "logs/${t}_test.out" 2>&1
  mkdir -p "results/smol_curve/$t" && cp "logs/$t/test_model_latest_hismap,siegfried.txt" "results/smol_curve/$t/"
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') gaps finished"

# ---- extra: de-confound the allocation comparison at ~5,100 tiles
# n-shot (cls_selection_100) scores 83.97 and corpus-class random scores 86.09. But the n-shot run
# has 18 sampler groups (9 pos / 9 neg, 50% positive), and every corpus-class run has 16
# (9 pos / 7 neg, ~56%). group_fixed_m:7:9 draws 63 positives and 63 negatives per epoch, which
# pins the corpus-class run at 50%. The railway and vineyard negative groups are still absent, and
# that part cannot be matched without changing the selection. Last in the queue, so it can be
# cancelled without losing a gap.
sel=budget_random_gap0_5087t_s2026
t=curve_nocap_5087t_r50
if [ ! -f "logs/$t/test_model_latest_hismap,siegfried.txt" ]; then
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" --sampler group_fixed_m:7:9 \
    --num-src 3 --src-selection retrieved > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" \
    --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
fi
mkdir -p "results/curve_own_prompts/$t" && cp "logs/$t/test_model_latest_hismap,siegfried.txt" "results/curve_own_prompts/$t/"
echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
echo "$(date '+%F %T') queue finished"
