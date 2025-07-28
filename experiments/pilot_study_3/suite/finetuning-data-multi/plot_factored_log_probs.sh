#!/usr/bin/env bash


python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/1000 \
 --name Nested-Oracle-100
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/1000 \
 --name Nested-Oracle-250
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/1000 \
 --name Nested-Oracle-500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/100 \
 --name Nested-Oracle-1000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/10 \
 --name Nested-Oracle-2500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/1 \
 --name Nested-Oracle-5000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/0.1 \
 --name Nested-Oracle-10000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/0.001 \
 --name Nested-Oracle-25000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0.001 \
 --name Nested-Oracle-50000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/100/1000 \
 --name No-Inference-100
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/250/1000 \
 --name No-Inference-250
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/500/1000 \
 --name No-Inference-500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/1000/100 \
 --name No-Inference-1000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/2500/10 \
 --name No-Inference-2500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/5000/10 \
 --name No-Inference-5000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/10000/1 \
 --name No-Inference-10000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/25000/1 \
 --name No-Inference-25000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/12/5000/0.0/1.0/50000/0.1 \
 --name No-Inference-50000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/100/10 \
 --name Regular-l24-h888-100
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/250/1 \
 --name Regular-l24-h888-250
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/500/1 \
 --name Regular-l24-h888-500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/1000/0.1 \
 --name Regular-l24-h888-1000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/2500/0.01 \
 --name Regular-l24-h888-2500
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/5000/1 \
 --name Regular-l24-h888-5000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/10000/0.1 \
 --name Regular-l24-h888-10000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/25000/0.1 \
 --name Regular-l24-h888-25000
python3 plot_factored_log_probs_single.py \
 --model /export/a02/artifacts/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/42/10/7/10/3/42/64/64/1e-5/24_888/10000/0.0/1.0/50000/0.01 \
 --name Regular-l24-h888-50000
