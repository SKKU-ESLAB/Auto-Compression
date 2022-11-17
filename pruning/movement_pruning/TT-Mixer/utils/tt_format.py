from __future__ import print_function

import torch
import torch.nn as nn

from torch import svd_lowrank as svd

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