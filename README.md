# Segment Your Notions Trought Retrieval Augmentation: a Cross-Domain Semantic-Aware segmentation model

## Installation
Using the `setup.sh` to create the conda environment for this project

## Prepare data
Using the converting tools in the package `data_tools` to convert your customized data into SYNTRA format. One example of converting the "trees" dataset for segmenting different spieces of trees are shown in `main.py`.

## Train
```bash
PYTHONPATH=. python syntra/train_one_gpu.py --config config_file.yaml
```

