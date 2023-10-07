#!/usr/bin/env bash

python3 generate_data.py \
    --seed 42 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type "interleaved" \
    --output_dir ../data/debug/train \
    --samples 1000 \