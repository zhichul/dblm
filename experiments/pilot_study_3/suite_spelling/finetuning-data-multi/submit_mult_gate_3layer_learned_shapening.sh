#!/usr/bin/env bash
source ~/.shtsrc
shopt -s expand_aliases
set -e
set -a

DEMO= #"-demo"
GROUP="finetune-data-multi"
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

for EVERY in 6 24
do
TRAIN_STEPS=1000
N=1000
INTERVAL=50
CUDA_VISIBLE_DEVICES=0
INFERENCE_EVERY="--inference_every_t ${EVERY}"
NAME_INFERENCE="-Inf-${EVERY}"
ts0 -L Mult-LearnSharpen-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_mult_gate.sh

TRAIN_STEPS=6000
N=10000
INTERVAL=500
CUDA_VISIBLE_DEVICES=0
INFERENCE_EVERY="--inference_every_t ${EVERY}"
NAME_INFERENCE="-Inf-${EVERY}"
ts0 -L Mult-LearnSharpen-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_mult_gate.sh

done
done
done
done
done
done
done
