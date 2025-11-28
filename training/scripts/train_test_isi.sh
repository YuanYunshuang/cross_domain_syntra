#!/usr/bin/env bash

CONFIG=trees_n4_full
DATADIR=/home/yuan/data/syntra/forest_monitoring/trees

# PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} --cache-data \
#   --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 --disable-runid

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} \
  --save-result --vis # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 


# CONFIG=trees_sep_n4_full
# DATADIR=/home/yuan/data/syntra/forest_monitoring

# PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} --cache-data \
#   # --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 --disable-runid

# PYTHONPATH=. python syntra/test.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} \
#   --save-result --vis # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 