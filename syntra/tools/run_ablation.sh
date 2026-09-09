#!/bin/bash
# SMOL -> SYNTRA, one variable at a time, at the ratio the sweep chose.
#
# Five runs on cls_selection_100 (5,087 tiles), everything held except the three things being
# tested. The model is identical throughout - model_name smol, SAM ViT-B, DoRA 16, example prompt
# encoder, default mask decoder, src_fusion max - as are the optimiser, schedule, loss, corpus,
# selection, pairings and the 37,935-pair test set. Only what the model is *shown* changes.
#
#              training           prompts        isolates
#   S0   smol dataset, k=1        pool draw      the published baseline
#   S1   cd_syntra2,   k=1        random         data organisation (S0 -> S1)
#   S1r  cd_syntra2,   k=1        retrieved      retrieval at one source (S1 -> S1r)
#   S2   cd_syntra2,   k=3        random         source count (S1 -> S2)
#   S3   cd_syntra2,   k=3        retrieved      retrieval at three sources (S2 -> S3)
#
# It is a 2x2 over {1,3} x {random, retrieved} with SMOL as the floor, so the source-count effect
# and the retrieval effect separate instead of confounding along a chain, and the interaction is
# visible - whether retrieval needs several sources or one good exemplar is enough.
#
# Ratio pinned at r* = 0.5 everywhere. The sweep measured 81.17 / 82.37 / 81.47 / 80.65 at 36 /
# 52 / 68 / 80% positive: a clean inverted-U peaking at 50%, and 1.72 points wide, well over the
# 0.7 MDE. cd_syntra2 gets 50% for free from 9 positive and 9 negative groups; SMOL needs
# pos_prob=0.5. SMOL's published 0.8 is the worst point on that curve, so tuning the *baseline*
# rather than our own method is what makes the comparison defensible - it costs us 1.7 points of
# apparent margin.
#
# src_fg_max 0.9 on every rung, matching SMOL's own `fg_ratio < v < 0.9`. Without it SYNTRA can
# prompt with near-total-coverage tiles that SMOL is denied - 14.1% of the pool overall, and 40%
# of forest's. Note this makes S3 a re-run of the anchor under the cap rather than the anchor
# itself.
#
# Ordered so the H2 result lands first: S1r trained on ONE retrieved source is the run whose
# test-time grid shows whether adding exemplars improves the model with no gradient step.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
SEL=cls_selection_100
SYN=syntra/syntra_anchor_cls100
SMOL=syntra/smol_budget
TRAIN="--data-info sample_pair_${SEL}_${SEL}_gap0"
TEST_INFO="sample_pair_${SEL}_test_samples_gap0"

run_syntra () {   # name, num_src, src_selection
  local t=$1 ns=$2 ss=$3
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$(date '+%F %T') $t done, skip"; return; }
  echo "$(date '+%F %T') === train $t (k=$ns, $ss)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SYN" -t "$t" -d "$D" \
    $TRAIN --num-src "$ns" --src-selection "$ss" --src-fg-max 0.9 > "logs/$t.out" 2>&1
  echo "$(date '+%F %T') === test  $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SYN" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
    --num-src "$ns" --src-selection "$ss" --src-fg-max 0.9 > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
}

run_smol () {
  local t=S0_smol_k1
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$(date '+%F %T') $t done, skip"; return; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SMOL" -t "$t" -d "$D" \
    $TRAIN --pos-prob 0.5 --source-file "${SEL}.json" --target-file "${SEL}.json" \
    > "logs/$t.out" 2>&1
  echo "$(date '+%F %T') === test  $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
    --source-file "${SEL}.json" > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
}

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting the ablation"

run_syntra S1r_k1_retrieved 1 retrieved
run_syntra S3_k3_retrieved  3 retrieved
run_syntra S1_k1_random     1 random
run_syntra S2_k3_random     3 random
run_smol

# Test-time grid: every trained model scored at k in {1,3,5,7}. Retrieval is baked into the
# pairing offline, so this costs no training - and for S1r it is the H2 result, whether a model
# trained on one exemplar improves when given more without a gradient step. Each variant writes
# to its own directory: the logger appends, so sharing one would stack blocks behind the
# canonical number, which is what makes the archived SMOL logs so easy to misread.
echo "$(date '+%F %T') === test-time grid"
for t in S1r_k1_retrieved S3_k3_retrieved S1_k1_random S2_k3_random; do
  ss=retrieved; case "$t" in *random) ss=random;; esac
  for ns in 1 3 5 7; do
    o="results/grid/${t}_ns${ns}"; mkdir -p "$o"
    [ -f "$o/test_model_latest_hismap,siegfried.txt" ] && continue
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SYN" \
      -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
      --num-src "$ns" --src-selection "$ss" --src-fg-max 0.9 -o "$o" > /dev/null 2>&1
    echo "$(date '+%F %T') grid $t k=$ns: $(grep -A1 'Overal Mean' "$o/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
  done
done
echo "$(date '+%F %T') ablation finished"
