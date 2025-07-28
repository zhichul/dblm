#!/usr/bin/env bash

for SEED in 42 # training experiment seed
do
for NVARS in 10
do
for NVALS in 7
do
for Z_SEED in 42
do
for SEQ_LEN in 10
do
for NBRANCHES in 3
do
for X_SEED in 42
do
for MEAN in 0.0
do
for STD in 1.0
do
for SAMPLE_SEED in 42
do
for NUM_SAMPLES in 500
do
TRAIN_STEPS=5000
GPU_BATCH_SIZE=20
GPU_NUM_SAMPLES=20
OUT_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/marginalized_seq2seq_models/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/12/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/
PRETRAINED_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/pretrained_seq2seq_models/42/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/64/64/1e-5/12/100000/checkpoint-100000/
mkdir -p ${OUT_DIR}

DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/data/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python3 ../../suite/finetuning/marginalize_encoder_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --dev_file ${DATA_FOLDER}/dev.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --pretrained_model ${PRETRAINED_DIR} \
    --seed ${SEED} \
    --logging_steps 200 \
    --gpu_batch_size ${GPU_BATCH_SIZE} \
    --output_dir ${OUT_DIR} \
    --project albo \
    --group finetune-data-multi \
    --name Marginalize-${NUM_SAMPLES} \
    --num_samples ${NUM_SAMPLES} \
    --gpu_num_samples ${GPU_NUM_SAMPLES} \
    --train_steps ${TRAIN_STEPS} \
    --position 9

done
done
done
done
done
done
done
done
done
done
done
