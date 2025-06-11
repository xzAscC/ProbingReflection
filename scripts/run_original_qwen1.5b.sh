python ./src/inference_vllm.py \
    --model_name "allenai/OLMo-2-0425-1B" \
    --dataset_name "HuggingFaceH4/MATH-500" \
    --output_dir "./asset/insert_response/" \
    --max_new_tokens 32784 \
    --dataset_ratio 1 \
    --do_sample \
    --use_cache \
