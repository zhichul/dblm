#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dl23


NUM_SAMPLES=500
SAMPLE_GPU_BATCH_SIZE=10
GPU_NUM_SAMPLES=20

PRETRAINED_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/spelling/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init/${SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${N}/${MULTI_LAMBDA}/checkpoint-early-stopping/
OUT_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/spelling/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init/${SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${N}/${MULTI_LAMBDA}/marginalized-result/
mkdir -p ${OUT_DIR}
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/spelling_data/default"
python3 ../../suite/finetuning/marginalize_encoder_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --dev_file ${DATA_FOLDER}/test.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --pretrained_model ${PRETRAINED_DIR} \
    --seed ${SEED} \
    --logging_steps 200 \
    --gpu_batch_size ${SAMPLE_GPU_BATCH_SIZE} \
    --output_dir ${OUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --gpu_num_samples ${GPU_NUM_SAMPLES} \
    --train_steps ${TRAIN_STEPS} \
    --project ${PROJ} \
    --group finteune-data-multi \
    --data_group spelling_same \
    --name VAE-${N}-Layer-${NLAYER}-${N}-${MULTI_LAMBDA}-Marg-${NUM_SAMPLES}${DEMO} \
    --shared_outcome_space
