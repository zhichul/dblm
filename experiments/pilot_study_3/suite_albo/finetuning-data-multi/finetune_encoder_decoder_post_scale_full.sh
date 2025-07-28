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
for BATCH_SIZE in 64
do
for GPU_BATCH_SIZE in 64
do
for LR in 1e-5
do
for NLAYER in 12
do
for TRAIN_STEPS in 5000
do
for SAMPLE_SEED in 42
do
for N in 100 250 500 1000 2500 5000 10000 25000 50000
do
for MULTI_LAMBDA in 0 0.001 0.01 0.1 1 10 100 1000
do
OUT_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}
PRETRAINED_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/pretrained_seq2seq_models/42/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/64/64/1e-5/12/100000/checkpoint-100000/
mkdir -p ${OUT_DIR}
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/data_bc/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
CUDA_VISIBLE_DEVICES=0 python3 ../../suite/finetuning/finetune_encoder_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --train_file ${DATA_FOLDER}/train.bin \
    --dev_file ${DATA_FOLDER}/dev.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --pretrained_model ${PRETRAINED_DIR} \
    --seed ${SEED} \
    --train_steps ${TRAIN_STEPS} \
    --eval_steps 200 \
    --logging_steps 200 \
    --learning_rate ${LR} \
    --gpu_batch_size ${GPU_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --n_layer ${NLAYER} \
    --output_dir ${OUT_DIR} \
    --project albo \
    --gating_mode albo_full \
    --group finetune-data-multi \
    --name PSG-Nested-Oracle-Full-${N}-${MULTI_LAMBDA} \
    --first ${N} \
    --multitask_lambda ${MULTI_LAMBDA} \
    --post_scale_gate


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
