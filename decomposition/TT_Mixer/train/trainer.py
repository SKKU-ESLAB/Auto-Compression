# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import logging

import numpy as np
import wandb
from tqdm import tqdm

from utils.tt_decomposition import TensorTrain
from utils.configs import CONFIGS
from utils.utils import *

from models.mlp_mixer import MlpMixer

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args):

        self.__set_configs(args)
        self.__set_model()
        self.__get_dataloader()
        
    def __set_configs(self, args):

        self.model_type = args.model_type

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        
        self.img_size = args.img_size
        self.num_classes = args.num_classes

        self.load_checkpoint = args.load_checkpoint
        self.load_path = args.load_path
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
        
        if args.target_layer == "all":
            self.target_layer = [i for i in range(12)]
        else:
            self.target_layer = [int(i) for i in args.target_layer.split(',')]
        
        hidden_tt_shape = [int(i) for i in args.hidden_tt_shape.split(',')]
        channels_tt_shape = [int(i) for i in args.channels_tt_shape.split(',')]
        tt_ranks = [int(i) for i in args.tt_ranks.split(',')]
        
        self.tt_configs = {768: hidden_tt_shape, 3072: channels_tt_shape, 'ranks': tt_ranks}
    
    def __init_admm(self):
        
        for name in self.admm_layers:
            param = self.model.get_parameter(name)
            self.Z.append(param.detach().cpu().clone())
            self.U.append(torch.zeros_like(param).cpu().clone())
            
        for name, param in self.model.named_parameters():
            if name not in self.admm_layers:
                if self.freeze_weights:
                    param.requires_grad = False
                    
    def __admm_loss(self, outputs, labels):
        loss = F.nll_loss(outputs, labels)
        for idx, name in enumerate(self.admm_layers):
            param = self.model.get_parameter(name)
            u = self.U[idx].to(self.device)
            z = self.Z[idx].to(self.device)
            
            loss += self.rho / 2 * (param - z + u).norm()
            
        return loss
            
    def __update_X(self):
        
        self.X = []
        for name in self.admm_layers:
            param = self.model.get_parameter(name)
            self.X.append(param.detach().cpu().clone())
            
    def __update_Z(self):

        self.Z = []
        for x, u in zip(self.X, self.U):
            z = x + u
            
    def __update_U(self):
        
        new_U = []
        for u, x, z in zip(self.U, self.X, self.Z):
            u = u + x - z
            new_U.append(u)
        self.U = new_U
        
    def __print_convergence(self, epoch):
        
        for idx, name in enumerate(self.admm_layers):
            x, z = self.X[idx], self.Z[idx]
            dis = torch.sqrt(torch.sum((x - z).square()))
            print("({}): {}".format(name, dis))
            wandb.log({{}: {}}.format(name, dis), step=epoch)
    
    def __set_admm_configs(self, args):
        
        self.epochs = args.admm_epochs
        self.rho = args.rho
        self.learning_rate = args.admm_learning_rate
        
    def __set_admm_layers(self):
        
        if self.use_incremental_learning:
            self.admm_layers = []
            self.admm_layers.append(self.target_name.pop())
            self.admm_layers.append(self.target_name.pop())
        else:
            self.admm_layers = self.target_name
            self.target_name = []
        
    def __set_model(self):
        
        config = CONFIGS[self.model_type]
        self.model = MlpMixer(config,
                              self.img_size,
                              self.num_classes,
                              patch_size=16,
                              zero_head=False).cpu()

        if self.load_checkpoint:
            self.__load_state_dict()
            
    def __set_tt_layers(self):

        for param in self.model.parameters():
            param.requires_grad = True

        if self.replaced_modules is None:
            return
        
        if self.use_incremental_learning:
            self.tt_layers = []
            self.tt_layers.append(self.target_name.pop())
            self.tt_layers.append(self.target_name.pop())
        else:
            self.tt_layers = self.target_name
            self.target_name = []

        if type(self.model) == nn.DataParallel:
            self.model, self.replaced_modules = replace_module_by_names(self.model.module,
                                                                        self.replaced_modules,
                                                                        self.tt_layers)
        else:
            self.model, self.replaced_modules = replace_module_by_names(self.model,
                                                                        self.replaced_modules,
                                                                        self.tt_layers)
            
            
        for name, param in self.model.named_parameters():
            if ("tt_cores" not in name) and self.freeze_weights:
                param.requires_grad = False
        
    def __load_state_dict(self):

        params = torch.load(self.load_path)
        self.model.load_state_dict(params)
            
        logger.info("Loading Model checkpoint from [DIR: {}]".format(self.load_path))
    
    def __save_state_dict(self):
        
        torch.save(self.model.state_dict(), self.save_path)
        
        logger.info("Saving Model checkpoint in [DIR: {}]".format(self.save_path))

    def __get_dataloader(self):

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        transform_test = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)

        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)

        self.train_loader = DataLoader(trainset,
                                       sampler=train_sampler,
                                       batch_size=self.train_batch_size,
                                       num_workers=4,
                                       pin_memory=True)
        self.test_loader = DataLoader(testset,
                                      sampler=test_sampler,
                                      batch_size=self.test_batch_size,
                                      num_workers=4,
                                      pin_memory=True) if testset is not None else None

    def __simple_accuracy(self, preds, labels):
        return (preds == labels).mean()

    def test(self, is_admm):
        self.model.eval()
        avg_loss = 0
        all_preds, all_labels = [], []
        
        epoch_iterator = tqdm(self.test_loader,
                        desc="Validating... (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
    
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                if is_admm:
                    test_loss = self.__admm_loss(F.log_softmax(output, dim=-1), labels)
                else:
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
        
        avg_loss = avg_loss / len(self.test_loader)
        top1 = self.__simple_accuracy(all_preds[0], all_labels[0])

        logger.info("Test Average Loss: %2.5f"  % avg_loss)
        logger.info("Test Top1 Accuracy: %2.5f" % top1)

        return top1        

    def train(self, is_admm):
        
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
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.cuda()
        
        self.model.zero_grad()
        
        for epoch in range(self.epochs):
            self.model.train()
            logger.info("Current Epoch: %d" %(epoch + 1))

            epoch_iterator = tqdm(self.train_loader,
                                  desc="Training (X / X Epochs) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            
            avg_loss = 0
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                if is_admm:
                    loss = self.__admm_loss(F.log_softmax(output, dim=-1), labels)
                else:
                    loss = F.nll_loss(F.log_softmax(output, dim=-1), labels)
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
            
                epoch_iterator.set_description(
                    "Training (%d / %d Epochs) (loss=%2.5f)" % (epoch, self.epochs, loss)
                )
                avg_loss += loss
            avg_loss = avg_loss / len(epoch_iterator)
            
            if is_admm:
                self.__update_X()
                self.__update_Z()
                self.__update_U()
                
                self.__print_convergence(epoch)
            
            acc = self.test(is_admm)
            wandb.log({"Loss": avg_loss, "Top1 Acc": acc},
                      step=epoch)
            
            if best_acc < acc:
                self.__save_state_dict()
                best_acc = acc

            if acc > 0.974:
                break
            
            lr_schedular.step()

    def fit(self, args):
        if self.train_type == "original_models":

            wandb.init(project="MLP-Mixer", entity="sminyu")
            wandb.config.update(args)
            wandb.watch(self.model)
            
            self.train(is_admm=False)
            
        elif self.train_type == "tt_models":
            self.replaced_modules = make_replaced_modules(self.model,
                                                          self.target_layer,
                                                          self.tt_configs)
            wandb.init(project="TT-Mixer", entity="sminyu")
            wandb.config.update(args)
            wandb.watch(self.model)

            self.target_name = make_target_name(self.model, self.target_layer)
            
            while self.replaced_modules is not None:
                self.model = self.model.cpu()
                self.__set_tt_layers()
                
                self.train(is_admm=False)
        
        if (self.train_type == "original_models") and self.use_admm:
            self.target_name = make_target_name(self.model, self.target_layer)
            self.__set_admm_configs(args)

            wandb.init(project="MLP-Mixer", entity="sminyu")
            wandb.config.update(args)
            wandb.watch(self.model)
            
            while self.target_name is not None:
                self.__set_admm_layers()
                self.__init_admm()

                self.train(is_admm=True)

