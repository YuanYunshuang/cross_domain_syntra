#!/usr/bin/env bash

CONFIG=trees_n4_full
DATADIR=/koko/datasets/syntra/forest_monitoring

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 