#!/usr/bin/env bash

for MODEL in interleaved # nested
do
for SEED in 11 12 13 14 15
do
python3 generate_data.py \
    --seed ${SEED} \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency/${MODEL}/${SEED}/train \
    --samples 10000


python3 generate_data.py \
    --seed ${SEED}0 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency/${MODEL}/${SEED}/dev \
    --samples 1000 \
    --use_predefined_model ../data/sample_efficiency/${MODEL}/${SEED}/train/ground_truth_models


python3 generate_data.py \
    --seed ${SEED}1 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency/${MODEL}/${SEED}/test \
    --samples 1000 \
    --use_predefined_model ../data/sample_efficiency/${MODEL}/${SEED}/train/ground_truth_models
done
done