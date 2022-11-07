# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import os

import torch
import torch.nn as nn
import numpy as np
import wandb


import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from utils.tt_format import TensorTrain
from utils.train import Trainer

from models.tt_mixer import TTMixer, CONFIGS
from models.mlp_mixer import MlpMixer
from models import configs

logger = logging.getLogger(__name__)

class TT_Trainer(Trainer):
    def __init__(self, args):
        self._set_model(args)
        self._set_train_configs(args)
        
    def _set_train_configs(self, args):

        if args.use_adam:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate,
                                        weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=args.learning_rate,
                                       momentum=0.9,
                                       weight_decay=args.weight_decay)
            
        if args.lr_schedular == "cosine":
            self.lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                     T_max=args.epochs,
                                                                     eta_min=0.001)
        else:
            self.lr_schedular = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                 gamma=0.95)
        
        self.epochs = args.epochs
        self.device = args.device
    
    def _set_target_layer(self, model):

        for name, param in model.named_parameters():
            name_list = name.split('.')
            if name_list[0] == "layer" and name_list[-1] == "weight":
                if name_list[2] == self.block_name:
                    if int(name_list[1]) not in self.target_layer:
                        self.preserved_list.append(name)
                    else:
                        self.target_list.append(name) 
                        
    def _load_preserved_weights(self, model, args):

        for name in self.preserved_list:
            preserved_param = model.get_parameter(name)
            param = self.model.get_parameter(name)
            if args.freeze_weights:
                preserved_param.requires_grad = False
            param.data = preserved_param
            
    def _load_target_weights(self, model, args):
        
        for name in self.target_list:
            target_param = model.get_parameter(name)
            decomp = TensorTrain(target_param.T,
                                 self.dict_shape[target_param.shape[1]],
                                 self.dict_shape[target_param.shape[0]],
                                 tt_ranks=args.tt_ranks)
            name = '.'.join(name.split('.')[:-1])
            for i in range(decomp.tt_dims):
                tt_core = name + '.tt_cores.{}'.format(i)
                param = self.model.get_parameter(tt_core)
                param.data = decomp.tt_cores[i]
    
    def _load_state_dict(self, args):
        super().set_model(args)
        
        self.target_layer = args.target_layer
        self.block_name = "channel_mlp_block"

        self.preserved_list = []
        self.target_list = []
        self.dict_shape = {768: args.hidden_tt_shape, 3072: args.channels_tt_shape}
        
        self._set_target_layer(super().model)
        
        self._load_preserved_weights(super().model, args)
        self._load_target_weights(super().model, args)
        
    
    def _set_model(self, args):

        config = configs.get_mixer_b16_tt_config(args)
        self.model = TTMixer(config,
                            args.img_size,
                            num_classes=args.num_classes,
                            patch_size=16,
                            zero_head=False,
                            target_layer=args.target_layer)
        self._load_state_dict(args)
        