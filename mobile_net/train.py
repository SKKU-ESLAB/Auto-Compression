import json
import os
import os.path as osp
import argparse
import math
import copy
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torchvision import transforms, datasets

from layers import MvPrunConv2d
from utils.utils import AverageMeter, accuracy, profile, replace_module_by_names, change_threshold
from model_zoo import ProxylessNASNets, MobileNetV3
from data_providers.imagenet import ImagenetDataProvider

class Trainer:
    def __init__(self, args):
        
        self.__set_configs(args)
        self.__set_model()
        self.__get_dataloader()

    def __set_configs(self, args):

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        
        self.img_size = args.img_size

        self.save_path = args.save_path

        self.train_type = args.train_type
        self.epochs = args.epochs
        
        self.device = args.device
        self.n_gpu = args.n_gpu
        
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.lr_schedular = args.lr_schedular
        self.max_grad_norm = args.max_grad_norm
        
        self.use_admm = args.use_admm
        self.freeze_weights = args.freeze_weights
        self.use_incremental_learning = args.use_incremental_learning
        
        self.warmup_steps = args.warmup_steps
        self.initial_warmup = args.initial_warmup
        self.final_warmup = args.final_warmup
        self.intial_threshold = args.initial_threshold
        self.final_threshold = args.final_threshold

    def __set_pruning_layers(self):
        
        if len(self.replaced_modules) == 0:
            return
        
        if self.use_incremental_learning:
            self.pruning_layers = []
            self.pruning_layers.append(self.target_name.pop())
        else:
            self.pruning_layers = self.target_name
            self.target_name = []
            
        if type(self.model) == nn.DataParallel:
            self.model, self.replaced_modules = replace_module_by_names(self.model.modules,
                                                                        self.replaced_modules,
                                                                        self.pruning_layers)
        else:
            self.model, self.replaced_modules = replace_module_by_names(self.model,
                                                                        self.replaced_modules,
                                                                        self.pruning_layers)            
    
    def __set_replaced_modules(self):

        self.replaced_modules = dict()
        self.target_name = []
        
        for name, module in self.model.named_modules():
            if (type(module) == nn.Conv2d) and module.kernel_size == (1, 1):
                    self.replaced_modules[name] = MvPrunConv2d(module)
                    self.target_name.append(name)
        
    def __set_model(self):
        
        net_config = json.load(open('net.config', 'r'))
        
        if net_config['name'] == ProxylessNASNets.__name__:
            model = ProxylessNASNets.build_from_config(net_config)
        elif net_config['name'] == MobileNetV3.__name__:
            model = MobileNetV3.build_from_config(net_config)
        else:
            raise ValueError("Not supproted network type: %s" % net_config['name'])
        
        init = torch.load("init", map_location="cpu")["state_dict"]
        model.load_state_dict(init)
        
    def __get_dataloader(self):
        
        data_provider = ImagenetDataProvider(train_batch_size=self.train_batch_size,
                                             test_batch_size=self.test_batch_size)
        
        self.train_loader = data_provider.train()
        self.test_loader = data_provider.test()
        
    def fit(self):
        
        self.__set_replaced_modules()
        
        num_layer = 0
        while len(self.replaced_modules) != 0:
            self.model.cpu()
            self.__set_pruning_layers()