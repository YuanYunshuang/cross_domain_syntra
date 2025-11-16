#!/usr/bin/env bash

WORKDIR=/home/yuang/data
DATADIR=${WORK_DIR}/syntra/autonomous_driving
CONFIG=cityscapes_n4_full


# Run training directly (replace args as needed)
docker run --gpus all --rm -it \
  -v "$WORKDIR":/workspace \
  -w /workspace \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  PYTHONPATH=${WORKDIR}/SYNTRA python ${WORKDIR}/SYNTRA/syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

# Run testing directly (replace args as needed)
docker run --gpus all --rm -it \
  -v "$WORKDIR":/workspace \
  -w /workspace \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  PYTHONPATH=${WORKDIR}/SYNTRA python ${WORKDIR}/SYNTRA/syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR}