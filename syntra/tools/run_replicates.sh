#!/bin/bash
# Replicate the headline, because right now it is a point estimate on top of ~1.0 of noise.
#
# S0_smol_k1 scored 81.36 and ratio_smol_p0.5 scored 82.37 on *identical* configuration - same
# data, same pos_prob, same 100k, same 37,935-pair test set - diverging from iteration 100. So the
# baseline is ambiguous by 1.01 and the headline margin is either +3.56 or +2.55 depending which
# is cited. Neither is defensible from one observation, and picking the lower baseline would be
# choosing the number that flatters us.
#
# It also means two results in the ablation are unsupported: S3-S2 = +0.62 and S3-S1r = +0.82 both
# sit inside the noise, so "retrieval helps on top of three sources" is not currently established.
# Three seeds on S3 and on S0 settle both questions at once.
#
# --seed varies the replicates deliberately, but note it does not pin them: DataLoader workers are
# forked sharing one numpy state and which worker serves which batch depends on timing, which is
# why the two runs above differ despite both using the default seed. Replicates would differ even
# without it; the flag just makes the variation intentional and recorded.
#
# Cost: 2 x ~12.5 h for S3 plus ~7 h for S0. S3 seed 1234 and both S0 observations already exist.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
SEL=cls_selection_100
SYN=syntra/syntra_anchor_cls100
SMOL=syntra/smol_budget
TRAIN="--data-info sample_pair_${SEL}_${SEL}_gap0"
TEST_INFO="sample_pair_${SEL}_test_samples_gap0"

while pgrep -f "train_one_gpu.py" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') GPU free, starting replicates"

for seed in 1235 1236; do
  t="S3_k3_retrieved_s${seed}"
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$t done, skip"; continue; }
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SYN" -t "$t" -d "$D" \
    $TRAIN --num-src 3 --src-selection retrieved --src-fg-max 0.9 --seed "$seed" \
    > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SYN" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
    --num-src 3 --src-selection retrieved --src-fg-max 0.9 > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done

t=S0_smol_k1_s1235
if [ ! -f "logs/$t/test_model_latest_hismap,siegfried.txt" ]; then
  echo "$(date '+%F %T') === train $t"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$SMOL" -t "$t" -d "$D" \
    $TRAIN --pos-prob 0.5 --source-file "${SEL}.json" --target-file "${SEL}.json" --seed 1235 \
    > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$SMOL" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "$TEST_INFO" \
    --source-file "${SEL}.json" > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
fi
echo "$(date '+%F %T') replicates finished"
