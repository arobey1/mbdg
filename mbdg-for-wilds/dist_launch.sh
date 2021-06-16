#!/bin/bash

# training parameters
export N_EPOCHS=5
export BATCH_SIZE=200
export LOG_INTERVAL=25
export TRAIN_ALG='MBDG-Reg'
export ARCH='densenet'
export OPTIMIZER='Adam'

# model
export MODEL_PATH=./core/models/learned_models/camelyon17/update/model.pt
export MUNIT_CONFIG_PATH=./core/models/learned_models/camelyon17/update/config.yaml

# dataset
export DATA_ROOT=../../domainbed/data
export DATASET='camelyon17'

# results
export RESULTS_PATH=./test-results
export LOG_PATH=./${RESULTS_PATH}/logs

# distributed settings
export N_GPUS_PER_NODE=4
export N_NODES=1

export OMP_NUM_THREADS=1

export learning_rates=(0.0001)
export lambda_dists=(0.1)
export num_steps=(1)
export num_trials=(1)

for n_steps in ${num_steps[@]}; do
    for lr in ${learning_rates[@]}; do
        for lam_dist in ${lambda_dists[@]}; do
            for t in ${num_trials[@]}; do
                python3 -m torch.distributed.launch \
                    --nproc_per_node=$N_GPUS_PER_NODE --nnodes=$N_NODES --node_rank=0 main.py \
                    --results-path $RESULTS_PATH --train-alg $TRAIN_ALG \
                    --n-epochs $N_EPOCHS --batch-size $BATCH_SIZE --log-interval $LOG_INTERVAL \
                    --dataset $DATASET --data-root $DATA_ROOT \
                    --model-path $MODEL_PATH --munit-config-path $MUNIT_CONFIG_PATH \
                    --distributed --trial-index $t --lr $lr --half-prec \
                    --mbdg-static-lam-dist $lam_dist --architecture $ARCH  \
                    --optim $OPTIMIZER --logging-root $LOG_PATH --pretrained \
                    --mbdg-num-steps $n_steps --mbdg-gamma 0.1 --mbdg-dual-step-size 0.05 
            done
        done
    done
done