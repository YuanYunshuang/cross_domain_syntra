#!/usr/bin/env bash

## Example of setting up shared folder that can modify the data written by docker
# mkdir -p ~/shared_acl
# # Allow everyone rwx, and make that the default for new files/dirs
# setfacl -m o:rwx ~/shared_acl
# setfacl -d -m o:rwx ~/shared_acl

# docker run --rm \
#   -u "$(id -u):$(id -g)" \
#   -v ~/shared_acl:/data \
#   IMAGE sh -lc 'umask 000 && exec your_command'


## Example usage:
# docker run --gpus all --rm -it \
#   -v /home/yuang/data:/workspace \
#   -w /workspace/cross_domain_syntra  \
#   --shm-size=16g \
#   opheliayuan/syntra:latest \
#   sh -lc "umask 000 && PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_cityscapes_n4_full.yaml -d ../syntra/autonomous_driving -o /workspace/logs"

export WORKDIR=/home/yuang/data
export DATA_SUBDIR=hismap
# export CONFIG=hismap_n1_full

for CONFIG in hameln_n1_full bayern_n1_full end; do
  if [ "$CONFIG" == "end" ]; then
      break
  fi
  # Run training directly (replace args as needed)
  docker run --gpus all --rm \
    -v "$WORKDIR":/workspace \
    -w /workspace/cross_domain_syntra \
    --shm-size=16g \
    opheliayuan/syntra:latest \
    sh -lc "umask 000 && PYTHONPATH=. python syntra/train_one_gpu.py \
    --config smol_${CONFIG}.yaml -d ../syntra -o /workspace/logs 
    # --cache-data 
    # --resume /workspace/logs/smol_${CONFIG}/model_latest.pth"

  # Run testing directly (replace args as needed)
  docker run --gpus all --rm \
    -v "$WORKDIR":/workspace \
    -w /workspace/cross_domain_syntra  \
    --shm-size=16g \
    opheliayuan/syntra:latest \
    sh -lc "umask 000 && PYTHONPATH=. python syntra/test.py \
    --config smol_${CONFIG}.yaml -d ../syntra -o /workspace/logs --checkpoint /workspace/logs/smol_${CONFIG}/model_latest.pth"
done
