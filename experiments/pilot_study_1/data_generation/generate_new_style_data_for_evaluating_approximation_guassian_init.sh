#!/usr/bin/env bash

for SEED in 11 12 13 14 15
do
mkdir -p ../data/new_style/evaluating_approximation_gaussian_init/${SEED}
python3 generate_new_style_data.py \
    --seed ${SEED} \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --z0_noise_weight 0.2 \
    --zt_noise_weight 0.2 \
    --model_type "interleaved" \
    --output_dir ../data/new_style/evaluating_approximation_gaussian_init/${SEED} \
    --samples 10000 \
    --z0_initializer_type gaussian \
    --z0_initializer_mean 0 \
    --z0_initializer_std 10 \
    --z0_initializer_min -60 \
    --z0_initializer_max 60 \
    --zt_initializer_type gaussian \
    --zt_initializer_mean 0 \
    --zt_initializer_std 10 \
    --zt_initializer_min -60 \
    --zt_initializer_max 60 \

done