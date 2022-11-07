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

    def _save_model(self, args):
        model_path = os.path.join("saved_models/tt_models", args.name + '.pt')
        torch.save(self.model.state_dict(), model_path)
        
        logger.info("Saving Model checkpoint in [DIR: {}]".format(model_path))
                        
    def _load_preserved_weights(self, model, args):

        for name in self.preserved_list:
            preserved_param = model.get_parameter(name)
            param = self.model.get_parameter(name)
            param.data = preserved_param
            if args.freeze_weights:
                param.requires_grad = False
            
    def _load_target_weights(self, model, args):
        
        for name in self.target_list:
            target_param = model.get_parameter(name)
            decomp = TensorTrain(target_param.T,
                                 self.dict_shape[target_param.shape[1]],
                                 self.dict_shape[target_param.shape[0]],
                                 tt_ranks=args.tt_ranks)
            decomp.fit()
            decomp.analyze_diff()
            name = '.'.join(name.split('.')[:-1])
            for i in range(decomp.tt_dims):
                tt_core = name + '.tt_cores.{}'.format(i)
                param = self.model.get_parameter(tt_core)
                param.data = decomp.tt_cores[i]
    
    def _tt_load_state_dict(self, args):
        model = super()._set_model(args).cpu()
        
        self.target_layer = args.target_layer
        self.block_name = "channel_mlp_block"

        self.preserved_list, self.target_list = self._set_target_layer(model)
        self.target_list = []
        self.dict_shape = {768: args.hidden_tt_shape, 3072: args.channels_tt_shape}
        
        self._set_target_layer(model)
        
        self._load_preserved_weights(model, args)
        self._load_target_weights(model, args)
        
    
    def _set_model(self, args):

        config = configs.get_mixer_b16_tt_config(args)
        self.model = TTMixer(config,
                            args.img_size,
                            num_classes=args.num_classes,
                            patch_size=16,
                            zero_head=False,
                            target_layer=args.target_layer).cpu()
        self._tt_load_state_dict(args)
        