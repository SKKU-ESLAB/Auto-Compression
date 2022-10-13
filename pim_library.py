import torch
import torch.nn as nn
import torch.nn.functional as F
import pim_cpp

"""
class PIM_BatchNorm1d(nn.Module):
    def __init__(self, in_features):
        super(PIM_BatchNorm1d, self).__init__()
        self.in_features = in_features
        self.scale = torch.randn(self.in_features)
        self.var = torch.randn(self.in_features)

    def forward(self, input):
        output = pim_cpp.bn_forward(input, self.scale, self.var)
        return output


class PIM_LSTM(nn.Module):
    def __init__(self, in_features, out_features):
        super(PIM_LSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_weight = torch.randn(self.out_features, out_features)
        self.input_weight = torch.randn(self.in_features, out_features)
        self.bias = torch.randn(self.out_features)

    def forward(self, input, hidden_state):
        output = pim_cpp.lstm_forward(input, hidden_state, self.hidden_weight, self.bias)
        return output
"""

class PIM_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PIM_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.ones(in_features, out_features)
        self.ukernel_code = 0
        self.pim_op_attrs = pim_cpp.PIM_OP_ATTRS()
        self.weight_ptr = 0

    def forward(self, input):
        output = pim_cpp.linear_forward(self.ukernel_code, input, self.weight_ptr, self.in_features, self.out_features)
        return output

def to_pim(model):
    parent = model
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            m = child.in_features
            n = child.out_features

            pim_op = pim_cpp.PIM_OP.GEMV
            
            if pim_cpp.isSuitableOps(pim_op, m, n):
                new_layer = PIM_Linear(m, n)
                new_layer.pim_op_attrs.GEMV(m, n)
                new_layer.ukernel_code = pim_cpp.GetMicrokernelCode(pim_op, new_layer.pim_op_attrs)
                new_layer.weight = torch.empty_like(child.weight).copy_(child.weight)
                new_layer.weight_ptr = pim_cpp.MapMemory(new_layer.weight, new_layer.ukernel_code, m*n)
                parent._modules[name] = new_layer

        elif isinstance(child, nn.Sequential):
            to_pim(child)

def init():
    print("PIM_INIT")
    pim_cpp.blas_init(0)
