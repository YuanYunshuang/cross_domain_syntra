#!/usr/bin/env bash

DATADIR=/home/yuan/data/syntra/hismap


PYTHONPATH=. python syntra/tools/inference_map.py test \
-d ${DATADIR}/hameln.3822_2023 \
-i ${DATADIR}/hameln.3822_2023/train_samples.txt \
-w /koko/SYNTRA_logs/smol/smol_hameln_n1_full/model_latest.pth \
-c smol/smol_hameln_n1_full.yaml \
-o /koko/SYNTRA_logs/smol/smol_hameln_n1_full/inference_results \
-n 1

