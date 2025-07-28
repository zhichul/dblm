#!/usr/bin/env bash
source ~/.shtsrc
shopt -s expand_aliases
set -e
set -a

DEMO= #"-demo"
GROUP="finetune-data-multi"
PROJ="colm-spelling"

for SEED in 42 44 46
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
CUDA_VISIBLE_DEVICES=1 
ts1 -L Mult-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_mult_gate_drop.sh

TRAIN_STEPS=6000
N=10000
INTERVAL=500
CUDA_VISIBLE_DEVICES=1 
ts1 -L Mult-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_mult_gate_drop.sh

done
done
done
done
done
done
