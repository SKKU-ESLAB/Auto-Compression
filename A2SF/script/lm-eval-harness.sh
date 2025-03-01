#!/bin/bash

CUDA=$1

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list openbookqa piqa arc_challenge arc_easy mathqa \
    --model_list meta-llama/Llama-2-7b-hf \
    --fewshot_list 1 3 5 \
    --ratio_list 0.1 0.2 0.3

# Dataset
## openbookqa piqa arc_challenge arc_easy mathqa

# Models
## meta-llama/Llama-2-7b-hf huggyllama/llama-7b facebook/opt-6.7b facebook/opt-2.7b