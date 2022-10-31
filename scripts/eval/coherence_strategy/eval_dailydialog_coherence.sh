#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

for seed in 123456 234567 345678 456789 567890; do
	for source in dailydialog_coherence_${seed}; do
		python run.py \
		    --eval_on fed-dialogue dstc9 persona-see \
		    --load_from output/train/coherence_base/${source} \
		    --output_dir output/coherence_base/${source} \
		    --model_name_or_path "roberta_full_base/" \
		    --seed ${seed}
	done
done
