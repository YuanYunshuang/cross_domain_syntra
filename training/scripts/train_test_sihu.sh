#!/usr/bin/env bash

CONFIG=remote_sensing_n4_s100
DATADIR=/koko/datasets/syntra/remote_sensing

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth