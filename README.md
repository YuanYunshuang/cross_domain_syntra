# Segment Your Notions Trought Retrieval Augmentation: a Cross-Domain Semantic-Aware segmentation model

## Updates
- [Nov 13] added data converting for kitti and cityscapes; loss function support for ignoring index 255; dataset support for multi-notion training; add dockerfile and pushed to dockerhub

## Version
- v1: dataset=cd_syntra
- v2: dataset=cd_syntra2_efficient
- v3: no dice loss for negative samples
- v4: use color gitter and rgb channel swap for augmentation, normalization at last

## Installation
Using the `setup.sh` to create the conda environment for this project

## Prepare data
Using the converting tools in the package `data_tools` to convert your customized data into SYNTRA format. One example of converting the "trees" dataset for segmenting different spieces of trees are shown in `main.py`.

## Train
```bash
PYTHONPATH=. python syntra/train_one_gpu.py --config config_file.yaml
```

## Where outputs live
Training runs write to `output_dir: logs` (checkpoints, `log.txt`, `config.yaml`, and the
`test_model_latest_*.txt` result files beside them), and the queue scripts in `syntra/tools/`
write their `logs/<task>.out` stdout there too. Since 2026-09-24 **`logs/` is a symlink to
`/koko/cd_syntra_logs_09_2026`** on the koko share (the local disk ran out; ~57 GB moved). Paths in
configs and scripts are unchanged - they still say `logs/...` - so nothing needs editing, but
`/koko` must be mounted (`/koko/mount_koko.sh`) before training or testing. If it is not, `logs`
is a dangling link and runs fail at start-up rather than writing somewhere else.

Older archives: `/koko/SYNTRA_logs/` (runs before the E13 branch, including v3).

`results/` stays in the repo: the small per-run score files and CSVs the docs cite.

1. label meaning
2. which labels are you using
3. how the evaluation metrics are calculated, e.g., IoU
4. confusing classes