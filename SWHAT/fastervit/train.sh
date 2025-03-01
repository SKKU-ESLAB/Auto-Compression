#!/bin/bash

DATA_PATH="/data/imagenet/"
MODEL=faster_vit_0_224
BS=1
EXP=Test
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py \
                --config configs/faster_vit_0_224_1k.yaml \
                --model faster_vit_0_224 \
                --tag faster_vit_0_224_exp_1 \
                --batch-size 256 \
                --drop-path 0.2 \
                --lr 0.00125 \
                --mesa 0.1 \
                --opt adamw \
                --weight-decay 0.005 \
                --amp \
                --input-size 3 224 224 \
                --data_dir ${DATA_PATH} \
                --log-wandb \
                --model-ema

