
#!/bin/bash

# training parameters
export N_EPOCHS=50
export BATCH_SIZE=32
export LOG_INTERVAL=25
export TRAIN_ALG='MBDG'
export ARCH='densenet'
export OPTIMIZER='Adam'

# model
export MODEL_PATH=./core/models/learned_models/camelyon17/large/model.pt
export MUNIT_CONFIG_PATH=./core/models/learned_models/camelyon17/large/munit.yaml

# dataset
export DATA_ROOT=./data
export DATASET='camelyon17'

# results
export RESULTS_PATH=./results
export LOG_PATH=./${RESULTS_PATH}/logs

export learning_rates=(0.01)
export lambda_dists=(0.5)
export num_steps=(1)

python3 main.py \
    --results-path $RESULTS_PATH --train-alg $TRAIN_ALG \
    --n-epochs $N_EPOCHS --batch-size $BATCH_SIZE --log-interval $LOG_INTERVAL \
    --dataset $DATASET --data-root $DATA_ROOT \
    --model-path $MODEL_PATH --munit-config-path $MUNIT_CONFIG_PATH \
    --trial-index 1 --lr 0.01 \
    --mbdg-static-lam-dist 1 --architecture $ARCH  \
    --optim $OPTIMIZER --logging-root $LOG_PATH --pretrained \
    --mbdg-num-steps 1 --mbdg-gamma 0.1 --mbdg-dual-step-size 0.1