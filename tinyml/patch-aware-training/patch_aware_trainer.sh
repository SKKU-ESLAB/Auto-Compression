torchrun --master_port 13444 --nproc_per_node=4 motiv_train.py --amp \
         --model efficientnet_b0 --epochs 20 --lr 0.0001 --wd 0.0001 --opt='rmsprop'\
         --weights="EfficientNet_B0_Weights.IMAGENET1K_V1" \
         --data-path '/data/imagenet/' -b 256  \
         --lr-scheduler="cosineannealinglr" \
	     --num-patches=4 --num-per-patch-stage=5 \
         --output-dir="efficient se correct zero" \
         --use-wandb
	 #--iters-to-accumulate=2 
         #--model mobilenet_v3_small --epochs 20 --lr 0.0004 --wd 0.0001 \
         #--weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1" \
         #--weights="ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1" \
         #--data-path '/data/imagenet/' -b 512  \
         #--model efficientnet_b0 --epochs 20 --lr 0.0004 --wd 0.0001 \
         #--weights="EfficientNet_B0_Weights.IMAGENET1K_V1" \
