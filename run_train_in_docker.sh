#!/usr/bin/env bash

WORK_DIR=/home/yuang/data


# Run training directly (replace args as needed)
WORK_DIR=/home/yuang/data && docker run --gpus all --rm -it \
  -v "$WORK_DIR":/workspace \
  -w /workspace \
  --shm-size=16g \
  opheliayuan/syntra:latest \
  PYTHONPATH=. python syntra/train_one_gpu.py \
  --config syntra_cityscapes_n4_s100.yaml -d ../syntra/autonomous_driving