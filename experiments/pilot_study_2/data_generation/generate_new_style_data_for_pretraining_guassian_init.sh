#!/usr/bin/env bash

for SEED in $(seq 11 20)
do
mkdir -p ../data/new_style/pretraining/${SEED}
python3 generate_new_style_data.py \
    --seed ${SEED} \
    --z0_num_variables 4 \
    --z0_num_values 4 \
    --sequence_length 5 \
    --z0_noise_weight 0.02 \
    --zt_noise_weight 0.02 \
    --model_type "interleaved" \
    --output_dir ../data/new_style/pretraining/${SEED} \
    --samples 1000000 \
    --z0_initializer_type gaussian \
    --z0_initializer_mean 0 \
    --z0_initializer_std 1 \
    --z0_initializer_min -6 \
    --z0_initializer_max 6 \
    --zt_initializer_type gaussian \
    --zt_initializer_mean 0 \
    --zt_initializer_std 1 \
    --zt_initializer_min -6 \
    --zt_initializer_max 6 \

done