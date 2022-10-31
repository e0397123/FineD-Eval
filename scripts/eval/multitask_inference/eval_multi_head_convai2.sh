export CUDA_VISIBLE_DEVICES=0
export dataset=convai2

for seed in 123456 234567 345678 456789 567890; do
    python run.py \
        --parallel_adapters \
        --multi_head \
        --eval_on fed-dialogue dstc9 persona-see \
        --train_on ${dataset}_coherence ${dataset}_likeable ${dataset}_nli \
        --load_from "output/train/multitask_base_${dataset}_${seed}" \
        --output_dir "output/multitask_base_${dataset}_${seed}" \
        --model_name_or_path "roberta_full_base" \
        --criterion loss --seed ${seed};
done
