#!/bin/bash

#SBATCH --job-name=train
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p new
#SBATCH -w ttnusa12

export CUDA_VISIBLE_DEVICES=3

for seed in "123456" "234567" "345678" "456789" "567890"; do
    for SRC in convai2_sentiment; do
        python run.py \
            --seed ${seed} \
            --train_on $SRC \
            --max_seq_len 256 \
            --epochs 3 \
            --max_train_examples 100000 \
            --max_dev_examples 10000 \
            --patience 10 \
            --learning_rate 1e-5 \
            --criterion "acc" \
            --bucket_sampler \
            --train_batch_size 32 \
            --eval_batch_size 32 \
            --eval_every 2000 \
            --full_eval_after_training \
            --save \
            --name "train/sentiment_base/${SRC}_${seed}" \
            --model_name_or_path "roberta_full_base/";
    done;
done;
