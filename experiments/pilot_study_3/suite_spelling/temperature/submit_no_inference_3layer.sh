#!/usr/bin/env bash
source ~/.shtsrc
shopt -s expand_aliases
set -e
set -a

DEMO= #"-demo"
GROUP="temperature"
PROJ="colm-spelling"

for SEED in 44 46 # 42
do
for BATCH_SIZE in 64
do
for GPU_BATCH_SIZE in 64
do
for LR in 1e-5
do
for NLAYER in 3
do
for MULTI_LAMBDA in 0
do

TRAIN_STEPS=1000
N=1000
INTERVAL=50
CUDA_VISIBLE_DEVICES=0
ts0 -L Noinf-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_no_inference.sh

TRAIN_STEPS=6000
N=10000
INTERVAL=500
CUDA_VISIBLE_DEVICES=0
ts0 -L Noinf-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_no_inference.sh

done
done
done
done
done
done