#!/usr/bin/env bash

CONFIG=trees_n4_s100
DATADIR=/koko/datasets/syntra

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} --vis-freq 100

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth