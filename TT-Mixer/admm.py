from __future__ import print_function
import logging
import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch

import torch.nn.functional as F
import numpy as np

from torch import svd_lowrank as svd
from tqdm import tqdm
from torchmetrics import Accuracy 

logger = logging.getLogger(__name__)

class TT_SVD:
    def __init__(self, original_weights: torch.Tensor, in_shapes: list, out_shapes: list, tt_ranks: list):
        
        self.weights = original_weights.cpu()
        self.in_features = self.weights.shape[0]
        self.out_features = self.weights.shape[1]
        
        self.in_shapes = in_shapes
        self.out_shapes = out_shapes
        self.ranks = [1] + tt_ranks + [1]
        self.tt_dims = len(in_shapes)
        
        assert self.tt_dims == len(self.ranks) - 1
        
        self.mat_to_ten()
        self.ten_to_tt()
        self.tt_to_ten()
        self.ten_to_mat()
    
    # Original weight to Tensorized weight for TT    
    def mat_to_ten(self):
        
        permute_shape = []
        self.tensorized_shape = []
        for i in range(self.tt_dims):
            permute_shape.append(i)
            permute_shape.append(self.tt_dims + i)
            self.tensorized_shape.append(self.in_shapes[i] * self.out_shapes[i])

        weight = self.weights.view(*self.in_shapes, *self.out_shapes)
        weight = torch.permute(weight, permute_shape)
        
        self.original_tensor = weight.reshape(*self.tensorized_shape)
        
    # Tensorized original weight to TT-format
    def ten_to_tt(self):
        tensor = self.original_tensor
        
        self.tt_cores = []
        for i in range(self.tt_dims -1):
            tensor = tensor.view(self.ranks[i]*self.tensorized_shape[i], -1)
            u, s, v = svd(tensor, q=self.ranks[i+1], niter=10)
            self.tt_cores.append(u.view(self.ranks[i], self.tensorized_shape[i], self.ranks[i+1]))

            tensor = torch.matmul(torch.diag(s), v.T)
            
        self.tt_cores.append(tensor.view(self.ranks[-2], self.tensorized_shape[-1], self.ranks[-1]))

        self.tt_weights = []
        for i in range(self.tt_dims):
            self.tt_weights.append(self.tt_cores[i].view(self.ranks[i], self.in_shapes[i], self.out_shapes[i], self.ranks[i+1]))
    
    # Reconstructing TT-format to Tensorized approximated weight        
    def tt_to_ten(self):
        tensor = self.tt_cores[0]
        for i in range(self.tt_dims -1):
            tensor = torch.matmul(tensor.reshape(-1, self.ranks[i+1]), self.tt_cores[i+1].reshape(self.ranks[i+1], -1))
        self.approx_tensor = tensor.view(*self.tensorized_shape)

    def ten_to_mat(self):
        
        permute_shape = []
        for i in range(self.tt_dims):
            permute_shape.append(2*i)
            
        for i in range(self.tt_dims):
            permute_shape.append(2*i + 1)
            
        temp = []    
        for i in range(self.tt_dims):
            temp.append(self.in_shapes[i])
            temp.append(self.out_shapes[i])
            
        temp_mat = self.approx_tensor.view(*temp)
        self.approx_weights = torch.permute(temp_mat, permute_shape).reshape(self.in_features, self.out_features)

    # Just testing approximation error
    def loss(self):
        criterion = nn.MSELoss()
        loss = criterion(self.weights, self.approx_weights)
        
        print("MSE loss: ", loss)


class TTAdmmTrainer:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device

        self.block_name = "channel_mlp_block"
        self.target_list = []
        
        for name, param in model.named_parameters():
            name_list = name.split('.')
            if name_list[0] == "layer" and name_list[-1] == "weight":
                if name_list[2] == self.block_name:
                    self.target_list.append(name)
            
        self.Z = []
        self.U = []
                
        for name in self.target_list:
            param = self.model.get_parameter(name)
            self.Z.append(param.detach().cpu().clone())
            self.U.append(param.detach().cpu().clone())
            
        self.dict_shape = {1024 : [16, 8, 8], 4096 : [16, 16, 16]}
        
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            filename='./admm.log',
                            level=logging.INFO)        
    
    def update_X(self, args):
        self.X = []
        for name in self.target_list:
            param = self.model.get_parameter(name)
            self.X.append(param.detach().cpu().clone())
    
    def update_Z(self, args):
        self.Z = []
        for x, u in zip(self.X, self.U):
            z = x + u
            tt_svd = TT_SVD(z,
                            in_shapes=self.dict_shape[z.shape[0]],
                            out_shapes=self.dict_shape[z.shape[1]],
                            tt_ranks=args.tt_ranks)
            self.Z.append(tt_svd.approx_weights)
            
    def update_U(self, args):
        self.U = []
        for u, x, z in zip(self.U, self.X, self.Z):
            u = u + x - z
            self.U.append(u)
            
    def admm_loss(self, output, target, args):
        loss = F.nll_loss(output, target)
        for idx, name in enumerate(self.target_list):
            param = self.model.get_parameter(name)
            u = self.U[idx].to(self.devices)
            z = self.Z[idx].to(self.devices)
            
            loss += args.rho / 2 * (param - z + u).norm()
        
        return loss
    
    def regularized_nll_loss(self, output, label, args):
        loss = F.nll_loss(output, label)
        if args.l2:
            for name in self.target_list:
                param = self.model.get_parameter(name)
                loss += args.alpha * param.norm()
        
        return loss
    
    def print_convergence(self):
        logger.info("Normalized Norm of (weight - projection)")
        for idx, name in enumerate(self.target_list):
            x, z = self.X[idx], self.Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
    
    def test(self, test_loader, args):
        self.model.eval()
        avg_loss = 0
        acc1, acc5 = 0, 0
        ACC1 = Accuracy()
        ACC5 = Accuracy(top_k=5)
        
        epoch_iterator = tqdm(test_loader,
                        desc="Validating... (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
    
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                test_loss = F.nll_loss(output, labels).item()
                avg_loss += test_loss
                acc1 += ACC1(output, labels)
                acc5 += ACC5(output, labels)

                epoch_iterator.set_description("Validating... (loss=%2.5f)" % test_loss)
            
        avg_loss /= len(test_loader.dataset)
        
        logger.info("Test Average Loss: %2.5f", avg_loss)
        logger.info("Test Top1 Accuracy: %2.5f %", (100. * acc1 / len(test_loader.dataset)))
        logger.info("Test Top5 Accuracy: %2.5f %", (100. * acc5 / len(test_loader.dataset)))

        return (100. * acc1 / len(test_loader.dataset))
    
    def warmup_train(self, train_loader, test_loader, optimizer, args):
        
        if args.decay_type == "cosine":
            scheduler = sch.CosineAnnealingLR(optimizer=optimizer,
                                              T_max=args.warmup_epochs,
                                              eta_min=0.0001)
        else:
            scheduler = sch.ExponentialLR(optimizer=optimizer,
                                          gamma=0.95)
            
        self.best_acc = 0
        
        for epoch in range(args.warmup_epochs):
            logger.info("Warm up Epoch: %d", (epoch + 1))
            self.model.train()
            epoch_iterator = tqdm(train_loader,
                                  desc="Warmup Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.regularized_nll_loss(output, labels)
                loss.backward()
                optimizer.step()
                
                epoch_iterator.set_description(
                    "Warmup Training (%d / %d Steps) (loss=%2.5f)" % (batch_idx, len(train_loader.dataset), loss)
                )
                
            acc = self.test(test_loader, args)
            if self.best_acc < acc:
                self.save_warmup_model(args)
                self.best_acc = acc
            
            scheduler.step()
        
    def admm_train(self, train_loader, test_loader, optimizer, args):
        self.best_acc = 0

        if args.decay_type == "cosine":
            scheduler = sch.CosineAnnealingLR(optimizer=optimizer,
                                              T_max=args.admm_epochs,
                                              eta_min=0.0001)
        else:
            scheduler = sch.ExponentialLR(optimizer=optimizer,
                                          gamma=0.95)        
        
        for epoch in range(args.admm_epochs):
            self.model.train()
            epoch_iterator = tqdm(train_loader,
                        desc="ADMM Training (X / X Steps) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
            logger.info("ADMM Epoch: %d", (epoch + 1))
            for batch_idx, (inputs, labels) in enumerate(epoch_iterator):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.admm_loss(output, labels, args)
                loss.backward()
                optimizer.step()

                epoch_iterator.set_description(
                    "Warmup Training (%d / %d Steps) (loss=%2.5f)" % (batch_idx, len(train_loader.dataset), loss)
                )
                
            self.update_X(args)
            self.update_Z(args)
            self.update_U(args)
            
            self.print_convergence()
            
            acc = self.test(test_loader, args)
            if self.best_acc < acc:
                self.save_admm_model(args)
                self.best_acc = acc

            scheduler.step()       
            
    def save_warmup_model(self, args):
        model_checkpoint = os.path.join("warmup_models", "%s.py" % args.name)
        torch.save(self.model.state_dict(), model_checkpoint)
        
        logger.info("Saved Warmup model checkpoint to [DIR: warmup_models]")
        
    def save_admm_model(self, args):
        model_checkpoint = os.path.join("admm_models", "%s.pt" % args.name)
        torch.save(self.model.state_dict(), model_checkpoint)
        
        logger.info("Saved ADMM model checkpoint to [DIR: admm_models]")