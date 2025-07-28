#!/usr/bin/env bash
source ~/.shtsrc
shopt -s expand_aliases
set -e
set -a

DEMO= #"-demo"
GROUP="finetune-data-multi"
PROJ="colm"

for SEED in 42 44 46
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
for NLAYER in 3
do
for SAMPLE_SEED in 42
do
for MULTI_LAMBDA in 0
do

TRAIN_STEPS=15000
N=20000
INTERVAL=1000
CUDA_VISIBLE_DEVICES=0 
ts0 -L Noinf-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_no_inference_dropout.sh

TRAIN_STEPS=20000
N=40000
INTERVAL=1000
CUDA_VISIBLE_DEVICES=0
ts0 -L Noinf-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_no_inference_dropout.sh

TRAIN_STEPS=25000
N=60000
INTERVAL=1000
CUDA_VISIBLE_DEVICES=0
ts0 -L Noinf-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_no_inference_dropout.sh

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
