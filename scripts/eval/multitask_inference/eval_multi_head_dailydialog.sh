export CUDA_VISIBLE_DEVICES=0
export dataset=dailydialog

for seed in 42; do
    python run.py \
        --parallel \
        --multi_head \
        --eval_on fed-dialogue dstc9 persona-see \
        --train_on ${dataset}_coherence ${dataset}_likeable ${dataset}_nli \
        --load_from "output/train/multitask_base_${dataset}_${seed}" \
        --output_dir "output/multitask_base_${dataset}_${seed}" \
        --model_name_or_path "roberta_full_base" \
        --criterion loss --seed ${seed};
done
