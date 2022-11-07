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
from utils.data_utils import get_loader

from models.mlp_mixer import MlpMixer, CONFIGS

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args):
        self.model = self._set_model(args)
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

        target_name = []
        for layer_idx in self.target_layer:
            target_name.append("layer.{}.{}.fc0.weight".format(layer_idx, self.block_name))
            target_name.append("layer.{}.{}.fc1.weight".format(layer_idx, self.block_name))
        
        target_list, preserved_list = [], []
        for name, param in model.named_parameters():
            if name in target_name:
                target_list.append(name)
            else:
                preserved_list.append(name)
                
        return target_list, preserved_list
    
    def _set_model(self, args):

        config = CONFIGS[args.model_type]
        model = MlpMixer(config,
                         args.img_size,
                         num_classes=args.num_classes,
                         patch_size=16,
                         zero_head=False)
        model = self._load_state_dict(model, args)
        return model
    
    def _save_model(self, args):
        model_path = os.path.join("saved_models/original_models", args.name + '.pt')
        torch.save(self.model.state_dict(), model_path)
        
        logger.info("Saving Model checkpoint in [DIR: {}]".format(model_path))
    
    def _load_state_dict(self, model, args):
        if os.path.exists(os.path.join("saved_models/original_models", 'Mixer-B_16_cifar_10_original.pt')):
            pretrained_path = os.path.join("saved_models/original_models", "Mixer-B_16_cifar_10_original.pt")
            logger.info("Loading Model checkpoint in [DIR: {}]".format(pretrained_path))
        else:
            pretrained_path = os.path.join("saved_models/pretrained_models/Mixer-B_16.pt")
            logger.info("Loading Model checkpoint in [DIR: {}]".format(pretrained_path))
        model.load_state_dict(torch.load(pretrained_path))

        return model
    
    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()
    
    def test(self, test_loader):
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
        logger.info("Test Top1 Accuracy: %2.5f" % (top1 * 100))

        return top1    
    
    def train(self, train_loader, test_loader, args):
        
        torch.cuda.empty_cache()
        self.best_acc = 0
        self.model.to(self.device)
        self.model.zero_grad()
        
        for epoch in range(args.epochs):
            logger.info("Epoch: {}".format(epoch + 1))
            self.model.train()
            epoch_iterator = tqdm(train_loader,
                                  desc="Training (X / X Epochs) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = F.nll_loss(F.log_softmax(output, dim=-1), labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                self.optimizer.step()
                
                epoch_iterator.set_description(
                    "Training (%d / %d Epochs) (loss=%2.5f)" % (epoch + 1, args.epochs, loss)
                )
                
            acc = self.test(test_loader)
            if self.best_acc < acc:
                self._save_model(args)
                self.best_acc = acc
            
            self.lr_schedular.step()
    
    def fit(self, args):
        train_loader, test_loader = get_loader(args)
        self.train(train_loader, test_loader, args)