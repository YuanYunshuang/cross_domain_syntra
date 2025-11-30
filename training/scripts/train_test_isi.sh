#!/usr/bin/env bash

CONFIG=trees_n4_full
DATADIR=/home/yuan/data/syntra/forest_monitoring/trees

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 