#!/bin/bash
# E-B selections and their pairings: random vs fps(+guard) at 300 and 1,000 tiles.
# Guard per rung from results/guard_tile_units_corpusclass.json: 15 at 300, 10 at 1,000.
# Random carries no guard - it *is* the density-neutral reference, and guarding it would
# change the control's candidate pool.
set -u
cd /home/yuan/code/CD_SYNTRA
PY=/home/yuan/anaconda3/envs/syntra/bin/python
D=/home/yuan/data/syntra
IDX=$D/global_index.npz
COMMON="-d $D --index $IDX -dkw hismap,siegfried --balance corpus-class --cell-gap 0 --budget-unit tile"

names=()
for spec in "300 15" "1000 10"; do
  set -- $spec; B=$1; G=$2
  if [ "$B" = "300" ]; then seeds="2026 2027 2028"; else seeds="2026"; fi
  for seed in $seeds; do
    for arm in random fps; do
      if [ "$arm" = "random" ]; then g=0; tag="budget_random_gap0_${B}t_s${seed}";
      else g=$G; tag="budget_fps_g${G}_gap0_${B}t_s${seed}"; fi
      echo "=== $tag"
      PYTHONPATH=. $PY data_tools/indexing/select_budget.py $COMMON \
        --budget "$B" --strategy "$arm" --density-guard "$g" --seed "$seed" --name "$tag" \
        2>&1 | grep -E "budget unit|pairs over|^    [a-z]"
      names+=("$tag")
    done
  done
done

echo
echo "########## pairings ##########"
for tag in "${names[@]}"; do
  for tgt in "$tag" test_samples; do
    echo "=== $tag -> $tgt"
    PYTHONPATH=. $PY data_tools/indexing/make_pool_pairing.py \
      -d $D --index $IDX --source "$tag" --target "$tgt" \
      -dkw hismap,siegfried --budget-unit tile 2>&1 \
      | grep -vE "^  !!" \
      | awk '/targets,/{gsub(",","",$2); gsub(",","",$4); t+=$2; p+=$4} \
             END{printf "  %d targets, %d target-class pairs\n", t, p}'
  done
done
echo "ALL DONE"
