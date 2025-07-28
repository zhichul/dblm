#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dl23

OUT_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/colm_finetuned_seq2seq_models_vary_data_multi_vae_random_init/${SEED}/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/${BATCH_SIZE}/${GPU_BATCH_SIZE}/${LR}/${NLAYER}/${TRAIN_STEPS}/${ZSEED}/${MEAN}/${STD}/${N}/${MULTI_LAMBDA}
PRETRAINED_DIR=/${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_3/pretrained_seq2seq_models/42/${NVARS}/${NVALS}/${SEQ_LEN}/${NBRANCHES}/${X_SEED}/64/64/1e-5/${NLAYER}/100000/checkpoint-100000/
NAME=VAE-Layer-${NLAYER}-${N}-${MULTI_LAMBDA}${DEMO}
mkdir -p ${OUT_DIR}
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/data/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
python3 ../../suite/finetuning/finetune_encoder_decoder.py \
    --data_config ${DATA_FOLDER}/args.json \
    --data_meta ${DATA_FOLDER}/meta.json \
    --train_file ${DATA_FOLDER}/train.bin \
    --dev_file ${DATA_FOLDER}/dev.bin \
    --test_file ${DATA_FOLDER}/test.bin \
    --pretrained_model ${PRETRAINED_DIR} \
    --seed ${SEED} \
    --train_steps ${TRAIN_STEPS} \
    --eval_steps ${INTERVAL} \
    --logging_steps ${INTERVAL} \
    --learning_rate ${LR} \
    --gpu_batch_size ${GPU_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --n_layer ${NLAYER} \
    --output_dir ${OUT_DIR} \
    --no_inference \
    --first ${N} \
    --multitask_lambda ${MULTI_LAMBDA} \
    --random_init \
    --multitask_data_source pilot_study_3_same \
    --vae \
    --project ${PROJ} \
    --group ${GROUP} \
    --name ${NAME}
