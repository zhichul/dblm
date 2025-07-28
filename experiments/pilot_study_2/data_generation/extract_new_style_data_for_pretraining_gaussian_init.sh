#!/usr/bin/env bash

for SEED in $(seq 11 20)
do
mkdir -p ../data/new_style/pretraining/${SEED}
python3 extract_new_style_data.py \
    --data_dir ../data/new_style/pretraining/${SEED}
head -n 990000 ../data/new_style/pretraining/${SEED}/samples/extracted_samples.jsonl > ../data/new_style/pretraining/${SEED}/samples/train.jsonl
tail -n 10000 ../data/new_style/pretraining/${SEED}/samples/extracted_samples.jsonl > ../data/new_style/pretraining/${SEED}/samples/dev.jsonl
done