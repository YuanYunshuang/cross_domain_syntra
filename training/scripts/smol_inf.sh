#!/usr/bin/env bash

DATADIR=/koko/datasets/syntra/hismap
SHEET=3922

for YEAR in 1898 1974 1982 1996; do

PYTHONPATH=. python syntra/tools/inference_smol.py \
-d ${DATADIR}/hameln.${SHEET}_${YEAR} \
-i train_samples.txt \
-w /koko/SYNTRA_logs/smol/smol_hameln_n1_full/model_latest.pth \
-c smol/smol_hameln_n1_full.yaml \
-o logs/inference_results 

done

