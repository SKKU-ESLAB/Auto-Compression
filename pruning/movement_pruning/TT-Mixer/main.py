# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np


import torch
import torch.nn as nn

from utils.data_utils import get_loader

from models.mlp_mixer import MlpMixer, CONFIGS
from models import configs
from utils.infer import Inference
from admm import TTAdmmTrainer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def set_model(args):
    config = CONFIGS[args.model_type]
    
    if args.dataset == "cifar_10":
        args.num_classes = 10
    elif args.dataset == "cifar_100":
        args.num_classes = 100
        
    model = MlpMixer(config, args.img_size, num_classes=args.num_classes, patch_size=16, zero_head=False)
    #model.load_state_dict(torch.load(os.path.join("saved_models/pretrained_models", args.model_type + ".pt")))
    model.load_state_dict(torch.load(os.path.join("saved_models/warmup_models", args.name + '.pt')))
    model.to(args.device)
    
    return model, args

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-type", choices=["Mixer-B_16", "Mixer-L_16"],
                        default="Mixer-B_16", help="Which model to use")
    parser.add_argument("--dataset", choices=["cifar_10, cifar_100"],
                        default="cifar_10", help="Which dataset to use")
    parser.add_argument("--train-batch-size", default=128, type=int,
                        help="Batch size for Training")
    parser.add_argument("--test-batch-size", default=128,
                        help="Batch size for Testing")
    
    parser.add_argument("--is-admm-training", default=0, type=int,
                        help="Training model if true")
    parser.add_argument("--use-adam", default=0, type=int,
                        help="Use ADAM for optimizer")
    parser.add_argument("--learning-rate", default=3e-2, type=float,
                        help="The initial learning rate for optimizer")
    parser.add_argument("--admm-learning-rate", default=6e-2, type=float,
                        help="The initial learning rate for optimizer")
    parser.add_argument("--target-layer", default=[0,1,2,3,4,5,6,7,8,9,10,11,12], nargs='+', type=int,
                        help="Indexes of layers that are applied with TT-format or Pruning")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some")
    parser.add_argument("--decay_type", choices=["cosine", "exp"], default="cosine",
                        help="How to decay the learning rate.")
    
    parser.add_argument("--warmup-training", default=0, type=int,
                        help="Begin with warmup training before ADMM training")
    parser.add_argument("--warmup-epochs", default=5, type=int,
                        help="Number of epochs for warmup training")
    parser.add_argument("--admm-epochs", default=30, type=int,
                        help="Number of epochs for ADMM training")
    
    parser.add_argument("--max-grad-norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--seed", default=77, type=int,
                        help="Random seed for initialization")
    
    parser.add_argument('--rho', type=float, default=1e-1,
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',
                    help='l2 norm weight (default: 5e-4)')
    parser.add_argument('--l2', default=False, action='store_true',
                        help='apply l2 regularization')

    parser.add_argument("--train-tt", default=0, type=int,
                        help="Finetuning TT-Mixer")
    parser.add_argument("--tt-ranks", default=[64, 64], type=list,
                        help="TT-ranks for TT-decomposition")
    parser.add_argument("--hidden-tt-shape", default=[8, 8, 12], type=list,
                        help="Factorized hidden dim shape for TT-format")
    parser.add_argument("--channel-tt-shape", default=[12, 16, 16], type=list,
                        help="Factorized channel dim shape for TT-format")
                    
    parser.add_argument("--inference", default=0, type=int,
                        help="Test model mode")
    parser.add_argument("--test-original", default=0, type=int,
                        help="Test only original model")
    parser.add_argument("--test-prun-model", default=0, type=int,
                        help="Test pruned model")
    parser.add_argument("--test-tt-model", default=0, type=int,
                        help="Test tt-model")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    args.name = args.model_type + '_' + args.dataset + '_' + str(args.train_batch_size)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.img_size = 224

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    
    # Set seed
    set_seed(args)
    
    model, args = set_model(args)
    
    train_loader, test_loader = get_loader(args)
    
    if args.is_admm_training:
        trainer = TTAdmmTrainer(model, args) 
        trainer.fit(train_loader, test_loader, args)

    if args.inference:
        infer = Inference(model, args)
        infer.test(test_loader, args)
        
    if args.train_tt:
        configs.get_mixer_b16_tt_config(args)
    
if __name__=="__main__":
    main()
