import json

from tqdm import tqdm
import numpy as np

import logging

import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from layers import MvPrunConv2d
from utils.utils import AverageMeter, accuracy, profile, replace_module_by_names, change_threshold
from model_zoo import ProxylessNASNets, MobileNetV3
from data_providers.imagenet import ImagenetDataProvider

logger = logging.getLogger(__name__)
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

        self.epochs = args.epochs
        
        self.device = args.device
        self.n_gpu = args.n_gpu
        
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.lr_schedular = args.lr_schedular
        self.max_grad_norm = args.max_grad_norm
        
        self.freeze_weights = args.freeze_weights
        self.use_incremental_learning = args.use_incremental_learning
        
        self.warmup_steps = args.warmup_steps
        self.initial_warmup = args.initial_warmup
        self.final_warmup = args.final_warmup
        self.initial_threshold = args.initial_threshold
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
            self.model, self.replaced_modules = replace_module_by_names(self.model.module,
                                                                        self.replaced_modules,
                                                                        self.pruning_layers)
        else:
            self.model, self.replaced_modules = replace_module_by_names(self.model,
                                                                        self.replaced_modules,
                                                                        self.pruning_layers)            
        
        for name, param in self.model.named_parameters():
            for target_name in self.pruning_layers:
                if (target_name not in name) and self.freeze_weights:
                    param.requires_grad = False
    
    def __set_threshold(self, m):
        if type(m) == MvPrunConv2d:
            if self.use_incremental_learning:
                if m in self.pruning_layers:
                    m.threshold = self.cur_threshold
            else:
                m.threshold = self.cur_threshold
    
    def __set_original_configs(self, lr, epochs):

        self.learning_rate = lr
        self.epochs = epochs
    
    def __set_adjustment_configs(self, lr, epochs):
        
        if type(self.model) == nn.DataParallel:
            self.model = self.model.module
            
        for param in self.model.parameters():
            param.requires_grad = True
            
        ori_lr, ori_epochs = self.learning_rate, self.epochs
        self.learning_rate = lr
        self.epochs = epochs
        
        return ori_lr, ori_epochs
    
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
            self.model = ProxylessNASNets.build_from_config(net_config)
        elif net_config['name'] == MobileNetV3.__name__:
            self.model = MobileNetV3.build_from_config(net_config)
        else:
            raise ValueError("Not supproted network type: %s" % net_config['name'])
        
        init = torch.load("init", map_location="cpu")["state_dict"]
        self.model.load_state_dict(init)
        
    def __get_dataloader(self):
        
        data_provider = ImagenetDataProvider(train_batch_size=self.train_batch_size,
                                             test_batch_size=self.test_batch_size,
                                             image_size=self.img_size)

        _transform = data_provider.build_train_transform()
        train_set = data_provider.train_dataset(_transform)
        test_set = data_provider.test_dataset(_transform)
        
        train_sampler = RandomSampler(train_set)
        test_sampler = SequentialSampler(test_set)
        
        self.train_loader = DataLoader(train_set,
                                       sampler=train_sampler,
                                       batch_size=self.train_batch_size,
                                       num_workers=4,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      sampler=test_sampler,
                                      batch_size=self.test_batch_size,
                                      num_workers=4,
                                      pin_memory=True)
                                        
    @staticmethod
    def schedule_threshold(
            step: int,
            total_step: int,
            warmup_steps: int,
            initial_threshold: float,
            final_threshold: float,
            initial_warmup: int,
            final_warmup: int):
        
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
        return threshold
    
    @staticmethod
    def onehot(label, n_classes):
        return torch.zeros(label.size(0), n_classes).scatter_(
            1, label.view(-1, 1), 1)
     
    def mixup(self, data, targets, alpha, n_classes):
        indices = torch.randperm(data.size(0))
        data2 = data[indices]
        targets2 = targets[indices]

        targets = self.onehot(targets, n_classes)
        targets2 = self.onehot(targets2, n_classes)

        lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
        data = data * lam + data2 * (1 - lam)
        targets = targets * lam + targets2 * (1 - lam)

        return data, targets
       
    def test(self, criterion):
        
        self.model.eval()
        test_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        epoch_iterator = tqdm(self.test_loader,
                        desc="Validating... (loss=X.X) (Top1=X.X) (Top5=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                test_loss = criterion(output, labels)
                test_losses.update(test_loss.item(), inputs.size(0))
                
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                
                epoch_iterator.set_description("Validating... (loss=%2.5f)" % (test_losses.avg))
                
        logger.info("Test Average Loss: %2.5f"  % test_losses.avg)
        logger.info("Test Top1 Accuracy: %2.5f" % top1.avg)
        
    def train(self):
        
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.learning_rate,
                              momentum=0.9,
                              weight_decay=self.weight_decay)

        if self.lr_schedular == "cosine":
            lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=self.epochs,
                                                                eta_min=0.001)
        else:
            lr_schedular = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=0.95)
            
        best_acc = 0
        
        torch.cuda.empty_cache()
        criterion = nn.CrossEntropyLoss().cuda()
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model).cuda()
        else:
            self.model.cuda()  
            
        self.model.zero_grad()
        
        for epoch in range(self.epochs):
            self.model.train()
            logger.info("Current Epoch: %d" %(epoch + 1))

            epoch_iterator = tqdm(self.train_loader,
                                  desc="Training (X / X Epochs) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            
            train_losses = AverageMeter()
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                current_step = batch_idx + (len(epoch_iterator) * epoch)
                total_step = len(epoch_iterator) * self.epochs
                self.cur_threshold = self.schedule_threshold(step=current_step,
                                                    total_step=total_step,
                                                    warmup_steps=self.warmup_steps,
                                                    final_threshold=self.final_threshold,
                                                    initial_threshold=self.initial_threshold,
                                                    final_warmup=self.final_warmup,
                                                    initial_warmup=self.initial_warmup,
                                                    )
                self.model.apply(self.__set_threshold)
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                #train_losses.update(loss.item(), inputs.size(0))
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
            
                epoch_iterator.set_description(
                    "Training (%d / %d Epochs) (loss=%2.5f)" % (epoch + 1, self.epochs, loss)
                )

            acc = self.test(criterion)

            if best_acc < acc:
                self.__save_state_dict()
                best_acc = acc
            
            lr_schedular.step()
        
    def fit(self):
        
        self.__set_replaced_modules()
        
        num_layer = 0
        while len(self.replaced_modules) != 0:
            self.model.cpu()
            self.__set_pruning_layers()

            logger.info("####### Current Layer: %d #######" %(num_layer))
            self.train()
            
            logger.info("####### Adjustment Stage #######")
            lr, epochs = self.__set_adjustment_configs(lr=(self.learning_rate/10),
                                                       epochs=1)
            self.train()
            self.__set_original_configs(lr=lr, epochs=epochs)
            num_layer += 1