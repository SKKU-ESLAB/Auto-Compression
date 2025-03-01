#!/bin/bash
DATA_PATH="/data/imagenet/val"
BS=128
#checkpoint='./weights/faster_vit_0.pth.tar'
checkpoint= '~/vit/swin_faster/output/train/faster_vit_0_224_exp_1/20241014-043559-faster_vit_0_224-224/model_best.pth.tar'

python validate.py --model faster_vit_0_224 --checkpoint=$checkpoint --data-dir=$DATA_PATH --batch-size $BS --input-size 3 224 224

