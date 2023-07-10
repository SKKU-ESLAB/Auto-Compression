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
        #MASK_TYPE=uvp2a
        #MASK_TYPE=uvp2u
        MASK_TYPE=unstructured
        ;;
    "5")
        #MASK_TYPE=uvp2u_cp
        MASK_TYPE=uvp2a_cp
        ;;
    "6")
        #MASK_TYPE=uvp4u
        MASK_TYPE=uvp4a
        ;;
    "7")
        #MASK_TYPE=uvp4u_cp
        MASK_TYPE=uvp4a_cp
        ;;
esac

DEVICE=$1

export TASK=mnli
# export TASK=qqp
export RECIPE=recipes/30epochs_unstructured90_glue_${MASK_TYPE}.yaml
# TASK can be either mnli or qqp

CUDA_VISIBLE_DEVICES=${DEVICE} python text_classification.py \
  --distill_teacher neuralmagic/oBERT-teacher-${TASK} \
  --model_name_or_path bert-base-uncased \
  --task_name ${TASK} \
  --do_train \
  --fp16 \
  --do_eval \
  --optim adamw_torch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --preprocessing_num_workers 8 \
  --seed 42 \
  --num_train_epochs 30 \
  --recipe ${RECIPE} \
  --output_dir output/bert-base/${TASK}/30e/90/${MASK_TYPE} \
  --overwrite_output_dir \
  --skip_memory_metrics true \
  --report_to wandb
