#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

for seed in 123456 234567 345678 456789 567890; do
	for source in convai2_order_${seed}; do
		python run.py \
		    --eval_on dstc9 fed-dialogue persona-see \
		    --load_from output/train/order_base/${source} \
		    --output_dir output/order_base/${source} \
		    --model_name_or_path "./roberta_full_base/" \
		    --seed ${seed}
	done
done
