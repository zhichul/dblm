#!/usr/bin/env bash


for SEED in 42
do
for NVARS in 10
do
for NVALS in  7
do
for SEQ_LEN in 10
do
for NBRANCHES in 3
do
for X_SEED in 42
do
for BATCH_SIZE in 64
do
for GPU_BATCH_SIZE in 64
do
for LR in 1e-5
do
for NLAYER in 3
do
for TRAIN_STEPS in 100000
do
OUT_DIR=${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/pretrained_seq2seq_models/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/
mkdir -p ${OUT_DIR}
CUDA_VISIBLE_DEVICES=1 python3 pretrain_encoder_decoder.py \
    --nvars ${NVARS} \
    --nvals ${NVALS} \
    --seq_len ${SEQ_LEN} \
    --n_branches ${NBRANCHES} \
    --x_model_seed ${X_SEED} \
    --seed ${SEED} \
    --train_steps ${TRAIN_STEPS} \
    --save_steps 10 \
    --logging_steps 1000 \
    --learning_rate ${LR} \
    --gpu_batch_size ${GPU_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --n_layer ${NLAYER} \
    --output_dir ${OUT_DIR} \

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
