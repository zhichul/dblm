#!/usr/bin/env bash

function bootstrap_shared_sample () {
    DATA_FOLDER=$1
    MODEL_DIR=$2
    K=$3
    GPU_BATCH_SIZE=16
    python3 ../../suite/finetuning/bootstrap_evaluate_encoder_decoder.py \
        --data_config ${DATA_FOLDER}/args.json \
        --data_meta ${DATA_FOLDER}/meta.json \
        --dev_file ${DATA_FOLDER}/dev.bin \
        --test_file ${DATA_FOLDER}/test.bin \
        --finetuned_model ${MODEL_DIR} \
        --gpu_batch_size ${GPU_BATCH_SIZE} \
        --post_scale_gate \
        --gating_mode shared_sample \
        --num_samples ${K}
}