
for SEED in $(seq 16 20)
do
    for N_LAYER in 1 2 3 4 5 6
    do  
        mkdir -p ${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_2/pretrained_seq2seq_models/${SEED}/${N_LAYER}/default/
        CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python3 pretrain.py \
            --seed ${SEED} \
            --data_config ../data/new_style/pretraining/${SEED}/ground_truth_models/config.json \
            --train_file ../data/new_style/pretraining/${SEED}/samples/train.jsonl \
            --dev_file ../data/new_style/pretraining/${SEED}/samples/dev.jsonl \
            --train_epochs 10 \
            --eval_epochs 1 \
            --learning_rate 1e-5 \
            --gpu_batch_size 512 \
            --batch_size 512 \
            --n_layer ${N_LAYER} \
            --output_dir ${BLU_ARTIFACTS2}/dblm/experiments/pilot_study_2/pretrained_seq2seq_models/${SEED}/${N_LAYER}/default/ \

    done
done
