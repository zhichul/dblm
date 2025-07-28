#!/usr/bin/env bash
source bootstrap/bootstrap_encoder_decoder_post_scale_full_sample.sh
source bootstrap/bootstrap_encoder_decoder_post_scale_full.sh
source bootstrap/bootstrap_encoder_decoder_post_scale_mult_gate.sh
source bootstrap/bootstrap_encoder_decoder_post_scale_shared_sample.sh
source bootstrap/bootstrap_encoder_decoder_post_scale.sh
source bootstrap/bootstrap_encoder_decoder_no_inference.sh


NVARS=10
NVALS=7
Z_SEED=42
X_SEED=42
SEQ_LEN=10
SAMPLE_SEED=42
NBRANCHES=3
MEAN=0.0
STD=1.0

export CUDA_VISIBLE_DEVICES=1

# ################################ full sample 500 ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/100/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/10/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/10/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/0.1/500
)
DTYPE="data_s"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_full_sample $DATA_FOLDER $MODEL/checkpoint-early-stopping/ 500
done

# ################################ shared sample 10 ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/10/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/1/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/0.1/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/0.1/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/0.01/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/0/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/0.001/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/0/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0/10
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_shared_sample $DATA_FOLDER $MODEL/checkpoint-early-stopping/ 10
done
# ################################ shared sample 100 ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/100/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/10/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/10/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/0.1/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/0.1/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/0.01/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/0.001/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/0.001/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0/100
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_shared_sample $DATA_FOLDER $MODEL/checkpoint-early-stopping/ 100
done
# ################################ shared sample 500 ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/100/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/10/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/10/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/0.1/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/0.1/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/0.01/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/0.001/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/0/500
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0/500
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_shared_sample $DATA_FOLDER $MODEL/checkpoint-early-stopping/ 500
done
# ################################ shared sample 1000 ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/100/100/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/250/10/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/500/10/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/1000/1/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/2500/0.1/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/5000/0.01/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/10000/0.01/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/25000/0/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/42/10/7/10/3/42/64/16/1e-5/12/5000/0.0/1.0/50000/0.001/1000
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_shared_sample $DATA_FOLDER $MODEL/checkpoint-early-stopping/ 1000
done
# ################################ ALBO-b ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_albo_b $DATA_FOLDER $MODEL/checkpoint-early-stopping/
done
# ################################ ALBO ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/0.1
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/0.1
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/0
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/0.01
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/0.001
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0.001
)
DTYPE="data_bc"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_albo $DATA_FOLDER $MODEL/checkpoint-early-stopping/
done
# ################################ MULT ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/1
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_mult $DATA_FOLDER $MODEL/checkpoint-early-stopping/
done
# ################################ No inference ################################ 
MODELS=(
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/1000
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/100
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/10
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/1
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/1
/export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0.1
)
DTYPE="data"
DATA_FOLDER="/home/blu/jhu/dblm/experiments/pilot_study_3/${DTYPE}/nvars=${NVARS}-nvals=${NVALS}-zseed=${Z_SEED}-seq_len=${SEQ_LEN}-nbranches=${NBRANCHES}-xseed=${X_SEED}-mean=${MEAN}-std=${STD}-sseed=${SAMPLE_SEED}-N=100000"
for MODEL in ${MODELS[@]}
do
    bootstrap_no_inference $DATA_FOLDER $MODEL/checkpoint-early-stopping/
done