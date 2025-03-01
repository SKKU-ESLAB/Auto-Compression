#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u run_text_generation.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --cache_budget 200 \
    --forgetting_factor 0.1