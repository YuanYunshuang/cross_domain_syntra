#!/usr/bin/env bash

CONFIG=trees_n4_full
DATADIR=/koko/datasets/syntra/forest_monitoring/trees

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} # --vis-freq 100 --disable-runid

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth