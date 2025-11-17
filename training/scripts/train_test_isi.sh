#!/usr/bin/env bash

CONFIG=trees_n4_s100
DATADIR=/home/yuan/data/syntra/forest_monitoring

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth