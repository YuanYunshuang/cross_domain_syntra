#!/usr/bin/env bash

WORKDIR=/home/yuang/data
DATA_SUBDIR=autonomous_driving
CONFIG=cityscapes_n4_full


# docker run --gpus all --rm -it \
#   -v /home/yuang/data:/workspace \
#   -w /workspace/cross_domain_syntra  \
#   --shm-size=16g \
#   opheliayuan/syntra:latest \
#   sh -lc "umask 000 && PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_cityscapes_n4_full.yaml -d ../syntra/autonomous_driving -o /workspace/logs"


# Run training directly (replace args as needed)
docker run --gpus all --rm \
  -v "$WORKDIR":/workspace \
  -w /workspace/cross_domain_syntra \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  sh -lc "umask 000 && PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_${CONFIG}.yaml -d ../syntra/${DATA_SUBDIR} -o /workspace/logs"

# Run testing directly (replace args as needed)
docker run --gpus all --rm -it \
  -v "$WORKDIR":/workspace \
  -w /workspace/cross_domain_syntra  \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  sh -lc "umask 000 && PYTHONPATH=. python syntra/test.py \
  --config syntra_${CONFIG}.yaml -d ../syntra/${DATA_SUBDIR} -o /workspace/logs"