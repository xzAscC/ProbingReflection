for alpha in 0.001 0.003 0.01 0.03 0.1 0.3 1.0
do
    python ./src/inference.py \
        --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        --dataset_name "HuggingFaceH4/MATH-500" \
        --output_dir "./asset/insert_response/" \
        --max_new_tokens 2048 \
        --dataset_ratio 0.1 \
        --do_sample \
        --use_cache \
        --injection \
        --injection_layer 20 \
        --injection_alpha $alpha
done