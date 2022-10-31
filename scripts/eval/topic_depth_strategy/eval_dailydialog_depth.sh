#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

for seed in 123456 234567 345678 456789 567890; do
	for source in dailydialog_nli_${seed} ; do
		python run.py \
			--max_seq_len 256 \
		    	--eval_on fed-dialogue dstc9 persona-see \
		    	--load_from output/train/topic_depth_base/${source} \
		    	--output_dir output/topic_depth_base/${source} \
		    	--model_name_or_path "roberta_full_base/" \
		    	--seed ${seed}
	done
done
