from __future__ import print_function

import torch
import torch.nn as nn

from torch import svd_lowrank as svd

class TensorTrain:
    def __init__(self, weights: torch.Tensor, row_shapes: list, col_shapes: list, tt_ranks: list):
        
        self.weights = weights.detach().cpu().clone()
        self.row_features = self.weights.shape[0]
        self.col_features = self.weights.shape[1]
        
        self.row_shapes = row_shapes
        self.col_shapes = col_shapes
        self.ranks = [1] + tt_ranks + [1]
        self.tt_dims = len(row_shapes)
        
        assert self.tt_dims == len(self.ranks) - 1
        
        self.match_dims()
        
    # Original Weight to TT-dimension Tensor   
    def match_dims(self):

        permute_shape = []
        self.tensor_shape = []
        for i in range(self.tt_dims):
            permute_shape.append(i)
            permute_shape.append(self.tt_dims + i)
            self.tensor_shape.append(self.row_shapes[i] * self.col_shapes[i])

        weights = self.weights.view(*self.row_shapes, *self.col_shapes)
        weights = torch.permute(weights, permute_shape)
        
        self.tensor = weights.reshape(*self.tensor_shape)

    def tt_svd(self):
        
        tensor = self.tensor
        
        self.tt_cores = []
        for i in range(self.tt_dims - 1):
            tensor = tensor.view(self.ranks[i] * self.tensor_shape[i], -1)
            U, S, V = svd(tensor, q=self.ranks[i+1], niter=20)
            self.tt_cores.append(U.reshape(self.ranks[i] * self.row_shapes[i],
                                           self.col_shapes[i] * self.ranks[i+1]))
            tensor = torch.matmul(torch.diag(S), V.T)
        self.tt_cores.append(tensor.reshape(self.ranks[-2] * self.row_shapes[-1],
                                            self.col_shapes[-1] * self.ranks[-1]))
        
    def tt_to_mat(self):
        
        row_permute, col_permute = [], []
        for i in range(self.tt_dims):
            row_permute.append(2 * i)
            col_permute.append(2 * i + 1)
        permute_shape = row_permute + col_permute
        
        tensor_shape = []
        for i in range(self.tt_dims):
            tensor_shape.append(self.row_shapes[i])
            tensor_shape.append(self.col_shapes[i])
            
        tensor = self.tt_cores[0].reshape(-1, self.ranks[1])
        for i in range(self.tt_dims - 1):
            tt_core = self.tt_cores[i + 1].reshape(self.ranks[i + 1], -1)
            tensor = torch.matmul(tensor, tt_core).reshape(-1, self.ranks[i + 2])

        tt_weights = tensor.view(*tensor_shape)
        self.tt_weights = torch.permute(tt_weights, permute_shape).reshape(self.row_features, self.col_features)
        
    def fit(self):
        
        self.tt_svd()
        self.tt_to_mat()
        
    def analyze_diff(self):
        
        norm_weights = self.weights.norm()
        norm_tt_weights = self.tt_weights.norm()
        dis_diff = torch.sqrt(torch.sum((self.weights - self.tt_weights).square()))
        direct_diff = torch.dot(self.weights.flatten(), self.tt_weights.flatten()) / (norm_tt_weights * norm_weights)
        
        print("Distance difference: {} & Direction difference: {}".format(dis_diff, direct_diff))

           
        
            
            
            
