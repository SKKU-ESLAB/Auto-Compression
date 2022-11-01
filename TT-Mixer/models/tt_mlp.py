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
            tt_core = dict(name="tt_core_%d" %(i),
                           shape=(self.tt_ranks[i], self.in_shapes[i], self.out_shapes[i], self.tt_ranks[i+1]))
            
            tmp = nn.Parameter(torch.Tensor(*tt_core["shape"]))
            self.register_parameter(tt_core["name"], tmp)
            
            tt_cores.append(tt_core)
        
        self.permute_shape = []
        for i in range(self.tt_dims + 2):
            if (i == 0) or (i == self.tt_dims + 1):
                self.permute_shape.append(i)
            elif i == self.tt_dims:
                self.permute_shape.append(1)
            else:
                self.permute_shape.append(i + 1)
        
        self.tt_info = dict()
        self.tt_info["tt_cores"] = tt_cores
        
    def tt_mat_mul(self, x):
        batch_size = x.shape[0]
        row_size = x.shape[1]

        # shape [batch_size x row_size, in_shape[0], ..., in_shape[-1]]
        x = x.view(-1, *self.in_shapes)

        '''
        multiply inputs & tt_core 0 ###
            input shape [batch_size x row_size, in_shape[1], ... , in_shape[-1], in_shape[0]]
            weight shape [in_shape[0], out_shape[0] * rank[1]]
        '''
        
        weight = getattr(self, "tt_core_0").view(self.in_shape[0], -1)
        x = torch.permute(x, self.permute_shape[:-1])
        x = torch.matmul(x, weight).view(-1, *self.in_shapes[1:], self.out_shapes[0], self.tt_ranks[1])
        
        for i in range(1, self.tt_dims):
            weight = getattr(self, "tt_core_%d" %(i)).view(self.tt_ranks[i] * self.in_shape[i], -1)
            x = x.view(-1, *self.in_shapes[i:], *self.out_shapes[:i], self.tt_ranks[i])
            x = torch.permute(x, self.permute_shape)
            x = torch.matmul(x, weight)
        
        x = x.view(batch_size, row_size, -1)
        
        return x
    
    def forward(self, x):
        x = self.tt_mat_mul(x)
        if self.bias is not None:
            x = torch.add(self.bias, x)
            
        return x
            
        