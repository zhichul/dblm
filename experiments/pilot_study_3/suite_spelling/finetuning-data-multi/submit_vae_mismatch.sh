#!/usr/bin/env bash
source ~/.shtsrc
shopt -s expand_aliases
set -e
set -a

DEMO= #"-demo"
GROUP="finetune-data-multi"
PROJ="colm-spelling"

for SEED in 42
do
for BATCH_SIZE in 64
do
for GPU_BATCH_SIZE in 64
do
for LR in 1e-5
do
for NLAYER in 3
do
for MULTI_LAMBDA in 1
do

# TRAIN_STEPS=30000
# N=357540
# INTERVAL=1000
# CUDA_VISIBLE_DEVICES=1

# TRAIN_STEPS=30000
# N=10000
# INTERVAL=1000
# CUDA_VISIBLE_DEVICES=1
# ts1 -L VAE-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_vae.sh
# ts1 -L Marg-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_marginalize_vae_mismatch.sh

TRAIN_STEPS=30000
N=1000
INTERVAL=1000
CUDA_VISIBLE_DEVICES=1
# ts1 -L VAE-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_vae.sh
ts1 -L Marg-S${SEED}-N${N}-T${TRAIN_STEPS} bash experiment_marginalize_vae_mismatch.sh

done
done
done
done
done
done
