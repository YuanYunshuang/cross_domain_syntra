#!/usr/bin/env bash

DATADIR=/home/yuan/data/syntra/hismap


for CONFIG in hismap_n1_s10 end; do
if [ "$CONFIG" = "end" ]; then
    break
fi
    
  PYTHONPATH=. python syntra/train_one_gpu.py \
  --config smol_${CONFIG}.yaml -d ${DATADIR}  
  --resume ./logs/smol_${CONFIG}/model_latest.pth # --vis-freq 100 

  PYTHONPATH=. python syntra/test.py \
  --config smol_${CONFIG}.yaml -d ${DATADIR} \
  --save-result --vis --checkpoint ./logs/smol_${CONFIG}/model_latest.pth 

done

