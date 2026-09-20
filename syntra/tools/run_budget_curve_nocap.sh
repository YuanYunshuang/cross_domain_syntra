#!/bin/bash
# The budget curve again, without the 90% source-coverage cap.
#
# The capped curve's first point came in at 66.38 against 76.24 for the same 300-tile selection
# trained without the cap - ten points, with recall flat in every class and precision collapsing
# (hameln.settlement 90 -> 51, hameln.grass 78 -> 57, railway 87 -> 73). The cap looks
# budget-dependent: at 5,087 tiles it helped, +0.95 mIoU and it fixed grass, but at 300 the prompt
# pool per (subset, class) is small enough that removing the >=90%-coverage exemplars strips the
# most prototypical examples and leaves marginal ones, so the model learns an over-inclusive
# notion of each class. `_cap_sources` only falls back when the cap empties a list outright, which
# does not trigger here.
#
# So the curve is run both ways and the pair is the result: whether src_fg_max has to be set per
# budget, exactly as section 6b found for the density guard. Everything else is identical to the
# capped runs - k=3, retrieved, r* = 0.5 from the 9+9 sampler groups, 100k, scored at k=3 on the
# same 37,935-pair test set - so the two curves differ in one flag.
#
# Waits for the capped round to finish rather than pre-empting it, so both curves are complete and
# comparable point for point.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
CFG=syntra/syntra_anchor_cls100
# Prompts must come from the run's OWN budget: at test time you only have the tiles you bought.
# Hardcoding cls_selection_100 here gave every budget the 5,087-tile prompt pool and cost ~10
# points at 300 tiles, which I first misattributed to the coverage cap.
TEST_INFO_TEMPLATE=1

while pgrep -f "run_budget_curve.sh" > /dev/null || pgrep -f "train_one_gpu.py" > /dev/null; do
  sleep 300
done
echo "$(date '+%F %T') capped round finished, starting the no-cap curve"

for b in 300 1000 3000; do
  sel="budget_random_gap0_${b}t_s2026"
  t="curve_nocap_${b}t"
  [ -f "logs/$t/test_model_latest_hismap,siegfried.txt" ] && { echo "$t done, skip"; continue; }
  echo "$(date '+%F %T') === train $t ($b tiles, no cap)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/train_one_gpu.py --config "$CFG" -t "$t" -d "$D" \
    --data-info "sample_pair_${sel}_${sel}_gap0" \
    --num-src 3 --src-selection retrieved > "logs/$t.out" 2>&1
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY syntra/test.py --config "$CFG" \
    -w "logs/$t/model_latest.pth" -d "$D" --data-info "sample_pair_${sel}_test_samples_gap0" \
    --num-src 3 --src-selection retrieved > "logs/${t}_test.out" 2>&1
  echo "$(date '+%F %T') === $t: $(grep -A1 'Overal Mean' "logs/$t/test_model_latest_hismap,siegfried.txt" 2>/dev/null | tail -1)"
done
echo "$(date '+%F %T') no-cap budget curve finished"
