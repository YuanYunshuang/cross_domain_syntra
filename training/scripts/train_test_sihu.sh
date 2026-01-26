#!/usr/bin/env bash

DATADIR=/koko/datasets/syntra/hismap


for CONFIG in hismap_n1_s100_v4 end; do
if [ "$CONFIG" = "end" ]; then
    break
fi
    
  PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR}  
  --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 

  PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} \
  --save-result --vis --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 

done