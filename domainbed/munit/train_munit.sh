#!/bin/bash

# Path to MUNIT configuration file.  Edit this file to change the number of iterations, 
# how frequently checkpoints are saved, and other properties of MUNIT.
# The parameter `style_dim` corresponds to the dimension of `delta` in our work.
export CONFIG_PATH=./core/tiny_munit.yaml

# Output images and checkpoints will be saved to this path.
export OUTPUT_PATH=./results-mnist

export CUDA_VISIBLE_DEVICES=0
python3 train_munit.py --config $CONFIG_PATH --output_path $OUTPUT_PATH