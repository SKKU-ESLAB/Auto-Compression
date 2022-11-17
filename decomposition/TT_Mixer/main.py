# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

import wandb
import torch
import torch.nn as nn

from train.trainer import Trainer

logger = logging.getLogger(__name__)

def set_configs(args):
    
    if args.load_model_type == "original_models":
        if args.use_admm:
            if args.target_layer == None:
                raise ValueError
            
            if (args.train_type == "tt_models"):
                load_model_name = "B-16_" + args.dataset + "_admm.pt"
                save_model_name = "B-16_" + args.dataset + "_admm.pt"
            elif (args.train_type != "tt_models"):
                load_model_name = "B-16_" + args.dataset + ".pt"
                save_model_name = "B-16_" + args.dataset + "_admm.pt"
        else:
            if (args.train_type == "tt_models"):
                load_model_name = "B-16_" + args.dataset + ".pt"
                save_model_name = "B-16_" + args.dataset + ".pt"
            elif (args.train_type != "tt_models"):
                load_model_name = "B-16_" + args.dataset + ".pt"
                save_model_name = "B-16_" + args.dataset + ".pt"
            
    elif args.load_model_type == "tt_models":
        assert args.train_type == "tt_models"
        
        if args.use_admm:
            load_model_name = "B-16_" + args.dataset + "_admm.pt"
            save_model_name = "B-16_" + args.dataset + "_admm.pt"
        else:
            load_model_name = "B-16_" + args.dataset + ".pt"
            save_model_name = "B-16_" + args.dataset + ".pt"            
                
    args.load_path = os.path.join(args.saved_path, args.load_model_type, load_model_name)
    args.save_path = os.path.join(args.saved_path, args.train_type, save_model_name)
    
    args.img_size = 224
    
    if args.dataset == "cifar_10":
        args.num_classes = 10
    elif args.dataset == "cifar_100":
        args.num_classes = 100
        
    if args.train_type == "original_models":
        args.model_type = "Mixer-B_16"
    elif args.train_type == "tt_models":
        args.model_type = "TT-Mixer-B_16"
    else:
        raise ValueError
    
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
    parser.add_argument("--dataset", choices=["cifar_10", "cifar_100"],
                        default="cifar_10",
                        help="Which dataset to use")
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
    parser.add_argument("--load-model-type",
                        choices=["original_models", "tt_models"],
                        default="original_models")
    parser.add_argument("--load-checkpoint", default=1,
                        type=int,
                        help="Loading model checkpoint from load path")
    
    # Training Arguments
    parser.add_argument("--train-type",
                        choices=["original_models", "tt_models"])
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
    
    # TT-format & Pruning Training Arguments
    parser.add_argument("--target-layer", default=None,
                        type=str,
                        help="Which layer of Model to be decomposed or pruned")
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
    
    # TT-format Configuration
    parser.add_argument("--tt-ranks", default="32, 32",
                        type=str,
                        help="Ranks for TT-Format")
    parser.add_argument("--hidden-tt-shape", default="8, 8, 12",
                        type=str,
                        help="Factorized hidden dimension for TT-format")
    parser.add_argument("--channels-tt-shape", default="12, 16, 16",
                        type=str,
                        help="Factorized channel dimension for TT-format")
    
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