# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.nn as nn

from tqdm import tqdm

import utils

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def set_model(args):
    config = CONFIGS[args.model_type]
    
    model = MlpMixer(config, args.img_size, num_classes=args.num_classes, patch_size=16, zero_head=True)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(args.device)
    
    num_params = utils.count_parameters(model)
    
    logger.info("{}".format(config))
    logger.info("Configurations: {}".format(args))
    logger.info("Total Parameters: \t%2.1fM" % num_params)
    return args, model

def main():
    parser = argparse.ArgumentParser()
    
    # Model & Directory
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16",
                                                 "Mixer-B_16-21k", "Mixer-L_16-21k"],
                        default="Mixer-B_16",
                        help="Which model to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/Mixer-B_16.npz",
                        help="Where to search for pretrained Mixer models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16",
                                                 "Mixer-B_16-21k", "Mixer-L_16-21k"],
                        default="Mixer-B_16",
                        help="Which model to use.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes to classify")
    parser.add_argument("--pretrained_dir", type=str, default="pretrained/Mixer-B_16.pt",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--save_dir", default="saved_models/", type=str,
                        help="The output directory where checkpoints will be written.")
    
    # Usage
    parser.add_argument("--inference_only", default=1, type=int,
                        help="Only inference will be processed if True")
    
    # Data size
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    
    # Training
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    # Distributed Training
    parser.add_argument("--gpu_idx", default="6,7", type=str,
                        help="GPU indices for distributed training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    # Setup Environments    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda:" + args.gpu_idx if torch.cuda.is_available() else "cpu")
    args.img_size = 224
    
    # Setup Logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    
    # Set Seed
    set_seed(args)    
    
    args, model = set_model(args)
    
    if args.inference_only:
        utils.test(model, args)
    else:
        utils.train(model, args)
        utils.test(model, args)
    
if __name__ == "__main__":
    main()
    

    
    