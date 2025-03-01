#!/bin/bash
DATA_PATH="/data/imagenet/val"
BS=128
checkpoint='./weights/faster_vit_0.pth.tar'

python swin_faster_val.py --model faster_vit_0_224 --checkpoint=$checkpoint --data-dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

