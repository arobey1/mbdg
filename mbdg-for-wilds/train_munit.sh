#!/bin/bash

export DATASET='camelyon17'
export DATA_DIR=../../training/data

# Path to MUNIT configuration file.  Edit this file to change the number of iterations, 
# how frequently checkpoints are saved, and other properties of MUNIT.
# The parameter `style_dim` corresponds to the dimension of `e` in our work.
export CONFIG_PATH=models/munit/core/big_munit.yaml

# Output images and checkpoints will be saved to this path.
export OUTPUT_PATH=./models/munit/results-camelyon

export CUDA_VISIBLE_DEVICES=0
python3 train_munit.py \
    --config $CONFIG_PATH --dataset $DATASET --output_path $OUTPUT_PATH \
    --data-root $DATA_DIR
    
