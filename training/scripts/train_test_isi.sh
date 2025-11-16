#!/usr/bin/env bash

CONFIG=hismap_n4_s100
DATADIR=/home/yuan/data/syntra/hismap

PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 