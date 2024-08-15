#/bin/bash

# $1: task
# $2: gpu number

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/noise_speckle_config.yaml \
    --gpu=7 \
    --save_dir=./results/mcg;
