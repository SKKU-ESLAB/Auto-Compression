from typing import Union

import torch
import torch.nn as nn

import numpy as np

from tt_module import _TTBase

class TTLinear(_TTBase):
    def __init__(self, in_features:int, out_features:int, ranks:Union[list, np.ndarray], dims:int, bias=False):
        super(TTLinear, self).__init__(in_features=in_features, out_features=out_features, ranks=ranks, dims=dims, bias=bias)

    def set_tt_cores(self):
        tt_cores_info = []
        for i in range(self.dims):
            tt_core_info = dict(
                    name="tt_core%d" % (i+1),
                    shape=(self.ranks[i], self.in_shape[i], self.out_shape[i], self.ranks[i+1]))
                    
            tmp = nn.Parameter(torch.Tensor(*tt_core_info["shape"]))
            self.register_parameter(tt_core_info["name"], tmp)

            tt_cores_info.append(tt_core_info)

        self.tt_info["tt_cores"] = tt_cores_info

    def set_params_info(self):
        
        original = self.in_features * self.out_features

        tt_format = np.sum(self.ranks[:self.dims] * self.in_shape * self.out_shape * self.ranks[1:self.dims +1])
        cr = original / tt_format

        self.tt_info["tt_format_params"] = tt_format
        self.tt_info["original_params"] = original
        self.tt_info["compression_ration"] = cr

        print("compression_ration is: ", cr)

    def tt_op(self, inputs):
        batch_size = inputs.shape[0]
        num_tokens = inputs.shape[1]
        inputs = inputs.view(-1, *self.in_shape)
        weight = getattr(self, "tt_core0")

        inputs = torch.permute(inputs, dims=[0, 2, 3, 1]).reshape(-1, self.in_shape[1] * self.in_shape[2], self.in_shape[0])
        x = torch.matmul(inputs, weight.view(-1, self.out_shape[0] * self.ranks[1]))
        
        weight = getattr(self, "tt_core1")
        x = x.reshape(x.shape[0], self.in_shape[1], self.in_shape[2], self.out_shape[0], self.ranks[1])
        x = torch.permute(x, dims=[0, 3, 2, 1, 4]).reshape(-1, self.out_shape[0] * self.in_shape[2], self.in_shape[1] * self.ranks[1])
        x = torch.matmul(res, weight.view(-1, self.out_shape[1] * self.ranks[2]))
        
        weight = getattr(self, "tt_core2")
        x = x.reshape(x.shape[0], self.out_shape[0], self.in_shape[2], self.out_shape[1], self.ranks[2])      
        x = torch.permute(x, dims=[0, 1, 3, 2, 4].reshape(-1, self.out_shape[0] * self.out_shape[1], self.in_shape[2] * self.ranks[2]))
        x = torch.matmul(x, weight.view(-1, self.out_shape[2] * self.ranks[3]))
        x = x.view(batch_size, num_tokens, -1)

        #for i in range(self.dims -1, 0, -1):
            #weight = getattr(self, "tt_core%d" %i)
            #res = torch.tensordot(res, weight, dims=([1, -1], [-2, 0]))
            #res = torch.einsum('bi...r, rkij-> b...kj', res, weight)

        return x

    def forward(self, inputs):
        x = self.tt_op(inputs)
        if self.bias is not None:
            x = torch.add(self.bias, x)

        return x
