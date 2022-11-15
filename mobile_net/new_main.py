# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import math
import json

import wandb
import torch
import torch.nn as nn

from train import Trainer

logger = logging.getLogger(__name__)

def set_configs(args):
    
    args.save_path = os.path.join(args.saved_path, args.train_type, save_model_name)
    
    if os.path.exists('run.config'):
        args.img_size = json.load(open('run.config'))['image_size']
    else:
        args.img_size = 224
    
    return args

def set_device(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_idx
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Total %s GPU are used (device idx: %s)" %(args.n_gpu, args.device_idx))
    
    return args

def set_seed(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
def main():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--train-batch-size", default=128,
                        type=int,
                        help="Batch size for training")
    parser.add_argument("--test-batch-size", default=128,
                        type=int,
                        help="Batch size for inference")
    parser.add_argument("--seed", default=77,
                        type=int,
                        help="Random seed for initialization")
    
    # Save & Load path
    parser.add_argument("--saved-path", default="saved_models")
    
    # Training Arguments
    parser.add_argument("--epochs", default=10,
                        type=int,
                        help="Number of epochs for training")
    parser.add_argument("--weight-decay", default=0,
                        type=int,
                        help="Weight decay if we apply some")
    parser.add_argument("--lr-schedular", choices=["cosine", "exp"],
                        default="cosine", help="Scheduling learning rate")
    parser.add_argument("--learning-rate", default=3e-2,
                        type=float,
                        help="Initial learning rate for optimizer")
    parser.add_argument("--max-grad-norm", default=1.0,
                        type=float,
                        help="Max gradient norm")
    
    # Pruning Training Arguments
    parser.add_argument("--use-incremental-learning", default=0,
                        type=int,
                        help="Using incremental learning if target layer is more than 1")
    parser.add_argument("--freeze-weights", default=0,
                        type=int,
                        help="Freeze weights from non-target layers")
    
    # ADMM Configuration
    parser.add_argument("--use-admm", default=0,
                        type=int,
                        help="Use ADMM alogorithm for TT-decomposition")    
    parser.add_argument("--admm-epochs", default=5,
                        type=int,
                        help="Number of epochs for ADMM training")
    parser.add_argument("--rho", default=1e-2,
                        type=float,
                        help="Cardinality weight in ADMM")
    
    # Pruning Configuration
    parser.add_argument("--warmup-steps", default=2,
                        type=int,
                        help="Warmup stage for movement pruning")
    parser.add_argument("--initial-warmup", default=5,
                        type=int,
                        help="Initial warmup for movement pruning")
    parser.add_argument("--final-warmup", default=5,
                        type=int,
                        help="Final warmup for movement pruning")
    parser.add_argument("--initial-threshold", default=1.0,
                        type=int,
                        help="Initial threshold for movement pruning (1 - sparsity)")  
    parser.add_argument("--final-threshold", default=0.25,
                        type=int,
                        help="Final threshold for movement pruning (1 - sparsity)")
        
    # CUDA Device Arguments
    parser.add_argument("--device-idx", default='0',
                        type=str,
                        help="Which GPU index to use")
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Additional Configurations
    args = set_device(args)
    args = set_configs(args)
    set_seed(args)
    
    train = Trainer(args)    
    train.fit(args)

if __name__=="__main__":
    main()