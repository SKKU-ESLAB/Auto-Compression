#!/bin/bash

mkdir -p pruned_model

sparsity=0.8
group_size=8

#for model in resnet20 resnet32 resnet44 renset56 resnet110 resnet1202
for model in resnet20 
do
    echo "python3 -u trainer.py --arch=$model  --save-dir=./pruned_model/$model_$group_size_$sparsity --pretrained"
    python3 -u pruner.py --arch=$model --save-dir=./pruned_model/"$model"_"$group_size"_"$sparsity" --pretrained --sparsity $sparsity --group_size $group_size
done
