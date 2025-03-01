#!/bin/bash

DATA_PATH="/data/imagenet/"
MODEL=faster_vit_0_224
BS=2
EXP=Test
LR=8e-4
WD=0.05
WR_LR=1e-6
DR=0.38
MESA=0.25

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 swin_faster.py --mesa ${MESA} --input-size 3 224 224 --crop-pct=0.875 \
--data_dir=$DATA_PATH --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR --warmup-lr $WR_LR --log-wandb
