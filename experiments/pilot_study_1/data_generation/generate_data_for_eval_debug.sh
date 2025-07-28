#!/usr/bin/env bash

python3 generate_data.py \
    --seed 43 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type "interleaved" \
    --output_dir ../data/debug/dev \
    --samples 100 \
    --use_predefined_model ../data/debug/train/ground_truth_models

python3 generate_data.py \
    --seed 44 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type "interleaved" \
    --output_dir ../data/debug/test \
    --samples 100 \
    --use_predefined_model ../data/debug/train/ground_truth_models
