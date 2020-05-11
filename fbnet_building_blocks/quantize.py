import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
                stochastic=False, inplace=False, num_chunks=None, out_half=False, quantize=False,  layer_num=-1, multi=False, index=[], is_act=False):
        if is_act:
            multi=False
        num_chunks = num_chunks = input.shape[
            0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            #min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            #max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
 
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if multi:
            bit_max = 8
            for i in range(bit_max):
                if len(index[layer_num][i]) == 0:
                    continue
                else:
                    idx = index[layer_num][i][0].tolist()
                min_value = output[idx].min()
                max_value = output[idx].max()
                qmin = 0.
                qmax = 2.**(1+i) - 1.
                scale = (max_value - min_value) / (qmax - qmin)
                scale = max(scale, 1e-8)
                output[idx] = output[idx].add_(-min_value).div_(scale).add_(qmin)
                output[idx] = output[idx].clamp_(qmin, qmax).round_()  # quantize
                output[idx] = output[idx].add_(-qmin).mul_(scale).add_(min_value)
        else:
            min_value = output.min()
            max_value = output.max()
            qmin = 0.
            qmax = 2.**num_bits - 1.
            scale = (max_value - min_value) / (qmax - qmin)
            scale = max(scale, 1e-8)
            output = output.add_(-min_value).div_(scale).add_(qmin)
            output = output.clamp_(qmin, qmax).round_()  # quantize
            output = output.add_(-qmin).mul_(scale).add_(min_value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None

def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False, quantize=False, layer_num=-1, multi=False, index=[], is_act=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, stochastic, inplace,  num_chunks, False, quantize, layer_num, multi, index, is_act)

class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, quantize=False, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits
        self.quantize = quantize

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16, quantize=self.quantize, is_act=True)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, num_bits=8, num_bits_weight=None, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, num_bits_grad=None, layer_num=-1, multi=False, index=[]):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits, quantize=False)
        if num_bits_weight != None:
            self.quantize = True
        self.layer_num = layer_num
        self.multi = multi
        self.index = index
    def forward(self, input):
        if self.quantize:
            #qinput = self.quantize_input(input)
            qinput = input
            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()), layer_num=self.layer_num, quantize=self.quantize, multi=self.multi, index=self.index)
            if self.bias is not None:
                qbias = quantize(self.bias, num_bits=self.num_bits_weight,quantize=False)
            else:
                qbias = None
            output = F.conv2d(qinput, qweight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False, quantize=False, layer_num=-1, multi=False, index=[]):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits, quantize=False)
        self.quantize = quantize
        self.layer_num = layer_num
        self.multi = multi
        self.index = index
    def forward(self, input):
        if self.quantize:
            #qinput = self.quantize_input(input)
            qinput = input
            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()), layer_num=self.layer_num, quantize=self.quantize, multi=self.multi, index=self.index)
            if self.bias is not None:
                qbias = quantize(self.bias, num_bits=self.num_bits_weight, quantize=False)
            else:
                qbias = None
            output = F.linear(qinput, qweight, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)
        return output

