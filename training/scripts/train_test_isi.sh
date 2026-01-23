#!/usr/bin/env bash

DATADIR=/home/yuan/data/syntra/hismap


for CONFIG in hismap_n1_s100_v2 end; do
if [ "$CONFIG" = "end" ]; then
    break
fi
    
  PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR}  
  --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 

  PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} \
  --save-result --vis --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 

  python syntra/evaluation/hismap.py \
  --result-folder ./logs/syntra_${CONFIG}/hismap \
  --data-folder ${DATADIR}
done




# CONFIG=trees_sep_n4_full
# DATADIR=/home/yuan/data/syntra/forest_monitoring

# PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} --cache-data \
#   # --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 --disable-runid

# PYTHONPATH=. python syntra/test.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} \
#   --save-result --vis # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 