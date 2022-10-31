#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export model=base

for seed in 123456 234567 345678 456789 567890; do
	for source in dailydialog_likeable_${seed} ; do
		python run.py \
			--max_seq_len 256 \
		    	--eval_on fed-dialogue dstc9 persona-see \
		    	--load_from output/train/likeable_${model}/${source} \
		    	--output_dir output/likeable_${model}/${source} \
		    	--model_name_or_path "roberta_full_${model}/" \
		    	--seed ${seed}
	done
done
