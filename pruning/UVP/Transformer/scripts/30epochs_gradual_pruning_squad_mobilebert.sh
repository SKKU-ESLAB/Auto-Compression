#!/bin/bash

case $1 in
    "0")
        MASK_TYPE=uvp2a
        ;;
    "1")
        MASK_TYPE=uvp2a_cp
        ;;
    "2")
        MASK_TYPE=uvp4a
        ;;
    "3")
        MASK_TYPE=uvp4a_cp
        ;;
    "4")
        MASK_TYPE=uvp2u
        ;;
    "5")
        MASK_TYPE=uvp2u_cp
        ;;
    "6")
        MASK_TYPE=uvp4u
        ;;
    "7")
        MASK_TYPE=uvp4u_cp
        ;;
esac

DEVICE=$1

export RECIPE=recipes/mobilebert_squad_e30_${MASK_TYPE}.yaml

# for 12-layer model: export MODEL=bert-base-uncased
# for 6-layer model: export MODEL=neuralmagic/oBERT-6-upstream-pretrained-dense
# for 3-layer model: export MODEL=neuralmagic/oBERT-3-upstream-pretrained-dense

CUDA_VISIBLE_DEVICES=${DEVICE} python question_answering.py \
    --distill_teacher zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
    --model_name_or_path zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
    --dataset_name squad \
    --do_train \
    --fp16 \
    --do_eval \
    --optim adamw_torch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --num_train_epochs 30 \
    --recipe ${RECIPE} \
    --output_dir output/mobilebert/squad/30e/obert_conf/${MASK_TYPE} \
    --overwrite_output_dir \
    --skip_memory_metrics true \
    --report_to wandb
