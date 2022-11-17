# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm

from utils.data_utils import get_loader
from utils.tt_format import TT_SVD

from models.tt_mixer import PrunMixer, TTMixer, CONFIGS
from models import configs

logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, original_model, args):
        self.device = args.device
        if args.test_original:
            logger.info("Testing MLP-Mixer")
            self.model = original_model
        else:
            logger.info("Testing MLP-Mixer where layer: {} is modified".format(args.target_layer))
            self.original_model = original_model
            self.target_layer = args.target_layer
            self.block_name = "channel_mlp_block"
            self.preserved_list = []
            self.target_list = []
        
            for name, param in original_model.named_parameters():
                name_list = name.split('.')
                if name_list[0] == "layer":
                    if name_list[2] == self.block_name:
                        if int(name_list[1]) not in self.target_layer:
                            self.preserved_list.append(name)
                        else:
                            self.target_list.append(name)
            self.set_target_model(args)
            
    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()
        
    def test(self, test_loader, args):
        self.model.to(self.device)
        self.model.eval()
        avg_loss = 0
        all_preds, all_labels = [], []
        
        epoch_iterator = tqdm(test_loader,
                        desc="Validating... (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
    
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                test_loss = F.nll_loss(F.log_softmax(output, dim=-1), labels).item()
                avg_loss += test_loss
                
                preds = torch.argmax(output, dim=-1)
                
                if len(all_preds) == 0:
                    all_preds.append(preds.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())
                else:
                    all_preds[0] = np.append(all_preds[0],
                                             preds.detach().cpu().numpy(), axis=0)
                    all_labels[0] = np.append(all_labels[0],
                                              labels.detach().cpu().numpy(), axis=0)

                epoch_iterator.set_description("Validating... (loss=%2.5f)" % test_loss)
        
        avg_loss = avg_loss / len(test_loader)
        top1 = self.simple_accuracy(all_preds[0], all_labels[0])

        logger.info("Test Average Loss: %2.5f"  % avg_loss)
        logger.info("Test Top1 Accuracy: %2.5f" % top1)

        return top1
        
    def prun_load_state_dict(self):
        for name in self.preserved_list:
            preserved_param = self.original_model.get_parameter(name)
            param = self.model.get_parameter(name)
            param.data = preserved_param
        
    def tt_load_state_dict(self, args):
        for name in self.preserved_list:
            preserved_param = self.original_model.get_parameter(name)
            param = self.model.get_parameter(name)
            param.data = preserved_param
            
        for name in self.target_list:
            target_param = self.original_model.get_parameter(name)
            tt_svd = TT_SVD(target_param, target_param.shape[1], target_param.shape[0], tt_ranks=args.tt_ranks)
            name = '.'.join(name.split('.')[:-1])
            for i in range(tt_svd.tt_dims):
                name = name + 'tt_core_{}'.format(i)
                param = self.model.get_parameter(name)
                param.data = tt_svd.tt_weights[i]
    
    def set_target_model(self, args):
        
        if args.dataset == "cifar_10":
            args.num_classes = 10
        elif args.dataset == "cifar_100":
            args.num_classes = 100
            
        if args.test_prun_model:
            config = CONFIGS[args.model_type]
            self.model = PrunMixer(config, args.img_size, args.num_classes, patch_size=16, zero_head=False, target_layer=self.target_layer)
            self.prun_load_state_dict()
        elif args.test_tt_model:
            config = configs.get_mixer_b16_tt_config(args)
            self.model = TTMixer(config, args.img_size, args.num_classes, patch_size=16, zero_head=False, target_layer=self.target_layer)
            self.tt_load_state_dict()
            
        