from typing import Union

import torch
import torch.nn as nn
import numpy as np

class TTLinear(nn.Module):
    def __init__(self, in_shapes, out_shapes, tt_ranks, bias=True):
        super(TTLinear, self).__init__()
        
        self.in_shapes = in_shapes
        self.in_features = np.prod(np.array(in_shapes))
        
        self.out_shapes = out_shapes
        self.out_features = np.prod(np.array(out_shapes))
        
        self.tt_ranks = [1] + tt_ranks + [1]
        self.tt_dims = len(in_shapes)
        
        self.checkinfo()
        
        self.set_tt_info()
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
            
    def checkinfo(self):
        assert len(self.tt_ranks) == (len(self.in_shapes) + 1), "tt_rank & input tt_shape does not match"

    def set_tt_info(self):
        tt_cores = []
        for i in range(self.tt_dims):
            tt_core = nn.Parameter(torch.Tensor(self.tt_ranks[i] * self.in_shapes[i], self.out_shapes[i] * self.tt_ranks[i + 1]))
            tt_cores.append(tt_core)
        self.tt_cores = nn.ParameterList(tt_cores)
        
        self.permute_shape = []
        for i in range(self.tt_dims + 2):
            if (i == 0) or (i == self.tt_dims + 1):
                self.permute_shape.append(i)
            elif i == self.tt_dims:
                self.permute_shape.append(1)
            else:
                self.permute_shape.append(i + 1)
        
    def tt_mat_mul(self, x):
        batch_size = x.shape[0]
        row_size = x.shape[1]

        # shape [batch_size x row_size, in_shape[0], ..., in_shape[-1]]
        x = x.view(-1, *self.in_shapes)

        '''
        Multiply inputs & tt_cores
            input shape [batch_size x row_size, in_shape[0], ... , in_shape[-1]]
            weight shape [rank[i] x in_shape[i], out_shape[i] x rank[i + 1]]
        '''
        
        weight = self.tt_cores[0]
        x = torch.permute(x, self.permute_shape[:-1])
        x = torch.matmul(x, weight)
        
        for i in range(1, self.tt_dims):
            weight = self.tt_cores[i]
            x = x.view(-1, *self.in_shapes[i:], *self.out_shapes[:i], self.tt_ranks[i])
            x = torch.permute(x, self.permute_shape)
            x = x.reshape(*x.shape[:-2], -1)
            x = torch.matmul(x, weight)
        
        x = x.view(batch_size, row_size, -1)
        
        return x
    
    def forward(self, x):
        x = self.tt_mat_mul(x)
        if self.bias is not None:
            x = torch.add(self.bias, x)
            
        return x