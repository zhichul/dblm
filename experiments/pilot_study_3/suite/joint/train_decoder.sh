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
for NLAYER in 0 1 2 3 6 12 24
do
for NHIDDEN in 768 384
do
for TRAIN_STEPS in 30000
do
for SAMPLE_SEED in 42
do
for N in 100000
do
OUT_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/joint_decoder_models/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${NHIDDEN}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/
mkdir -p ${OUT_DIR}
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/data/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=${N}"
CUDA_VISIBLE_DEVICES=1 python3 train_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --train_file ${DATA_FOLDER}/train.bin \
    --dev_file ${DATA_FOLDER}/dev.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --seed ${SEED} \
    --train_steps ${TRAIN_STEPS} \
    --eval_steps 1000 \
    --logging_steps 200 \
    --learning_rate ${LR} \
    --gpu_batch_size ${GPU_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --n_layer ${NLAYER} \
    --n_hidden ${NHIDDEN} \
    --output_dir ${OUT_DIR} \
    --group joint \
    --name Regular-${NLAYER}-${NHIDDEN}
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