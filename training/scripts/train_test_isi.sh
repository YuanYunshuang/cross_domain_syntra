#!/usr/bin/env bash

CONFIG=trees_n4_full
DATADIR=/home/yuan/data/syntra

# PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} --cache-data

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth