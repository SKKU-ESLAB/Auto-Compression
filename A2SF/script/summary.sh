#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u summary_test.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --cache_budget 250 \
    --data_path data/xsum-5shot.jsonl \
    --output_path results/original_5shot_250