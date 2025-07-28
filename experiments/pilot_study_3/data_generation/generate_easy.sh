#!/usr/bin/env bash

for NVARS in 10 #20
do
for NVALS in 7 # 5 6 7 # 2 3 4
do
for Z_SEED in 42
do
for SEQ_LEN in 10 #20 # 10 
do
for NBRANCHES in 2
do
for X_SEED in 42
do
for MEAN in 0.0
do
for STD in 1.1 #1.1 1.5 1.3 1.2 1.4 #1.0 #0.9 1.0 1.1 # 0.7 1.2 1.3 # 0.9 1.0 1.1
do
for SAMPLE_SEED in 42 # 43 44 45 # 46 47 48 49 50 51 
do
for N in 100000
do
# # z model
# NVARS=10
# NVALS=4
# Z_SEED=42

# # x model
# SEQ_LEN=10
# NBRANCHES=3
# X_SEED=42

# # Gaussian
# MEAN=0.0
# STD=1.0
MIN="-10"
MAX="10"

# # Sampling and IO
# SAMPLE_SEED=42
# N=10
SAVE_FOLDER="../data/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=${N}"

mkdir -p ${SAVE_FOLDER}
python3 generate_data.py \
    --nvars ${NVARS} \
    --nvals ${NVALS} \
    --initializer_mean ${MEAN} \
    --initializer_std ${STD} \
    --initializer_min ${MIN} \
    --initializer_max ${MAX} \
    --z_model_seed ${Z_SEED} \
    --seq_len ${SEQ_LEN} \
    --n_branches ${NBRANCHES} \
    --x_model_seed ${X_SEED} \
    --sample_seed ${SAMPLE_SEED} \
    --N ${N} \
    --save_name ${SAVE_FOLDER} \
    --never_top \
    --segments 0 \
    --segment_length 100000 
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
