#!/bin/bash

yes | pip install numpy matplotlib seaborn pandas scipy tqdm PyYAML

export TORCH='1.7.0'
export CUDA='cu110'
yes | pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
yes | pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
yes | pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
yes | pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
yes | pip install torch-geometric
yes | pip install wilds

git clone https://github.com/NVIDIA/apex
cd apex
yes | conda run -n stabenv pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./