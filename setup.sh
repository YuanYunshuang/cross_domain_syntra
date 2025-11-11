#!/usr/bin/env bash
set -euxo pipefail

conda create -y -n syntra python=3.10
# faiss (from listed channels)
conda run -n syntra conda install -y -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=11.4,<=11.8'
# torch and xformers
conda run -n syntra pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.29
# other requirements
conda run -n syntra pip install -r requirements.txt
