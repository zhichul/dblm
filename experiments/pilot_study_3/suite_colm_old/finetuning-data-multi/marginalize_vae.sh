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
for LR in 1e-5
do
for NLAYER in  3
do
for TRAIN_STEPS in 4000
do
for GPU_BATCH_SIZE in 64
do
for BATCH_SIZE in 64
do
for N in 500
do
for MULTI_LAMBDA in 1
do
for NOISE in "" #0.0 # 0.4
do


SAMPLE_GPU_BATCH_SIZE=20
GPU_NUM_SAMPLES=20

#PRETRAINED_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init_noise/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}/${NOISE}/checkpoint-early-stopping/
#PRETRAINED_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}/${NOISE}/checkpoint-early-stopping/
PRETRAINED_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init_resample/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}/${NOISE}/checkpoint-early-stopping/
OUT_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}/${NOISE}/marginalized-result/
mkdir -p ${OUT_DIR}
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/data/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python3 ../../suite/finetuning/marginalize_encoder_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --dev_file ${DATA_FOLDER}/test.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --pretrained_model ${PRETRAINED_DIR} \
    --seed ${SEED} \
    --logging_steps 200 \
    --gpu_batch_size ${SAMPLE_GPU_BATCH_SIZE} \
    --output_dir ${OUT_DIR} \
    --project colm \
    --group finetune-data-multi-2k \
    --name Random-Init-VAE-${N}-${MULTI_LAMBDA}-Marg-${NUM_SAMPLES} \
    --num_samples ${NUM_SAMPLES} \
    --gpu_num_samples ${GPU_NUM_SAMPLES} \
    --train_steps ${TRAIN_STEPS} \
    #--noise 0.99
    #--noise 0.99

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
done
done
done
done
done
done
done
done
