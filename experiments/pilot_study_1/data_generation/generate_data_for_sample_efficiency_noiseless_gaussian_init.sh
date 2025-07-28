#!/usr/bin/env bash

for MODEL in interleaved # nested
do
for SEED in 11 12 13 14 15
do
mkdir -p ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/train
python3 generate_data_noiseless.py \
    --seed ${SEED} \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/train \
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


mkdir -p ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/dev
python3 generate_data_noiseless.py \
    --seed ${SEED}0 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/dev \
    --samples 1000 \
    --use_predefined_model ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/train/ground_truth_models \
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

mkdir -p ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/test
python3 generate_data_noiseless.py \
    --seed ${SEED}1 \
    --z0_num_variables 8 \
    --z0_num_values 5 \
    --sequence_length 10 \
    --noise_weight 0.2 \
    --model_type ${MODEL} \
    --output_dir ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/test \
    --samples 1000 \
    --use_predefined_model ../data/sample_efficiency_noiseless_gaussian_init/${MODEL}/${SEED}/train/ground_truth_models \
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
done
