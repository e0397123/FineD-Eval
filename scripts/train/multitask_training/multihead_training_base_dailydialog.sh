#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export dataset=dailydialog

for seed in 42; do

python run.py \
    --train_on ${dataset}_coherence ${dataset}_likeable ${dataset}_nli \
    --max_seq_len 256 \
    --seed ${seed} \
    --epochs 3 \
    --max_train_examples 100000 \
    --max_dev_examples 10000 \
    --learning_rate 1e-5 \
    --adapter_learning_rate 1e-5 \
    --criterion "acc" \
    --multi_head \
    --save \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --eval_every 2000 \
    --full_eval_after_training \
    --name "train/multitask_base_${dataset}_${seed}" \
    --patience 10 \
    --model_name_or_path "roberta_full_base";

done;
