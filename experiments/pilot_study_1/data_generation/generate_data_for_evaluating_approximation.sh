#!/usr/bin/env bash

for SEED in 11 12 13 14 15
do
python3 generate_data.py \
    --seed ${SEED} \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type "interleaved" \
    --output_dir ../data/evaluating_approximation/${SEED} \
    --samples 10000

done