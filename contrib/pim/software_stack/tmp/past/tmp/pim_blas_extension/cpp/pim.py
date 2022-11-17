import torch
from torch import nn
import pim_cpp

class PIM_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PIM_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.ones(2,2)

    def forward(self, input):
        output = pim_cpp.linear_forward(input, self.weight)
        return output


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


