# Segment Your Notions Trought Retrieval Augmentation: a Cross-Domain Semantic-Aware segmentation model

## Updates
- [Nov 13] added data converting for kitti and cityscapes; loss function support for ignoring index 255; dataset support for multi-notion training; add dockerfile and pushed to dockerhub

## Installation
Using the `setup.sh` to create the conda environment for this project

## Prepare data
Using the converting tools in the package `data_tools` to convert your customized data into SYNTRA format. One example of converting the "trees" dataset for segmenting different spieces of trees are shown in `main.py`.

## Train
```bash
PYTHONPATH=. python syntra/train_one_gpu.py --config config_file.yaml
```

