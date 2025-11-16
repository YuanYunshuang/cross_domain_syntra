#!/usr/bin/env bash

WORKDIR=/home/yuang/data
DATADIR=${WORK_DIR}/syntra/autonomous_driving
CONFIG=cityscapes_n4_full


docker run --gpus all --rm \
  -v /home/yuang/data:/workspace \
  -w /workspace/cross_domain_syntra  \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  bash -c "PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_cityscapes_n4_full.yaml -d ../syntra/autonomous_driving"


# Run training directly (replace args as needed)
docker run --gpus all --rm \
  -v "$WORKDIR":/workspace \
  -w /workspace/SYNTRA  \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  python ${WORKDIR}/SYNTRA/syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR} 

# Run testing directly (replace args as needed)
docker run --gpus all --rm -it \
  -v "$WORKDIR":/workspace \
  -w /workspace/SYNTRA  \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  python ${WORKDIR}/SYNTRA/syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ${DATADIR}