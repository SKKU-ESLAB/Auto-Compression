from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np 
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lvs):    
        return input.mul(n_lvs-1).round_().div_(n_lvs-1)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Q_ReLU(nn.Module):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.n_lvs = 0
        self.bits = 32
        self.act_func = act_func
        self.inplace = inplace
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.theta = Parameter(torch.Tensor(0))

    def initialize(self, bits, offset, diff):
        if isinstance(bits, list):
            n_lvs = [2**i for i in bits]
        elif isinstance(bits, int) and bits != 32 and bits != 0:
            n_lvs = [2**bits]
        else:
            n_lvs = [self.n_lvs]
        
        if len(n_lvs)>1:
            self.n_lvs = Tensor(n_lvs)
            self.bits = Tensor(bits)
        else:
            self.n_lvs = n_lvs[0]
            self.bits = bits

        print(n_lvs)
        print(self.n_lvs)
        print(bits)
        print(self.bits)  

        self.theta = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if x.dim() > 2:
            N, C, H, W = x.shape
            act_size = H * W
        else:
            N1, N2 = x.shape
            act_size = N2
            
        if self.act_func:
            x = F.relu(x, self.inplace)
        
        if isinstance(self.n_lvs, int) and self.n_lvs == 0:
            act_size *= 32
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = F.hardtanh(x / a, 0, 1)
            
            if not isinstance(self.n_lvs, torch.Tensor):
                x = RoundQuant.apply(x, self.n_lvs) * c
                act_size *= self.n_lvs
                return x
            else:
                # 1) for loop
                softmask = F.gumbel_softmax(self.theta, tau=1, hard=False)
                softmask = softmask
                x_bar = torch.zeros_like(x)

                for i, n_lv in enumerate(self.n_lvs):
                    x_bar += RoundQuant.apply(x, n_lv) * c * softmask[i]
                
                act_size *= (softmask * self.n_lvs).sum()
                
                return x_bar

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, bits, offset, diff):
        if isinstance(bits, list):
            n_lvs = [2**i for i in bits]
        elif isinstance(bits, int) and bits != 32 and bits != 0:
            n_lvs = [2**bits]
        else:
            n_lvs = [self.n_lvs]
        
        if len(n_lvs)>1:
            self.n_lvs = Tensor(n_lvs)
            self.bits = Tensor(bits)
        else:
            self.n_lvs = n_lvs[0]
            self.bits = bits

        self.theta = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.n_lvs = 0
        self.bits = 32
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.theta = Parameter(torch.Tensor(0))

    def initialize(self, bits, offset, diff):
        if isinstance(bits, list):
            n_lvs = [2**i for i in bits]
        elif isinstance(bits, int) and bits != 32 and bits != 0:
            n_lvs = [2**bits]
        else:
            n_lvs = [self.n_lvs]
        
        if len(n_lvs)>1:
            self.n_lvs = Tensor(n_lvs)
            self.bits = Tensor(bits)
        else:
            self.n_lvs = n_lvs[0]
            self.bits = bits

        self.theta = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))


    def forward(self, x):
        N, C, H, W = x.shape
        if isinstance(self.n_lvs, int) and self.n_lvs == 0:
            act_size = 32 * H * W
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = F.hardtanh(x / a, -1, 1)

            if not isinstance(self.n_lvs, torch.Tensor):
                x = RoundQuant.apply(x, self.n_lvs // 2) * c
                act_size = self.n_lvs * H * W
                return x
            else:
                softmask = F.gumbel_softmax(self.theta, tau=1, hard=False)
                softmask = softmask
                x_bar = torch.zeros_like(x)
                
                for i, n_lv in enumerate(self.n_lvs):
                    x_bar += RoundQuant.apply(x, n_lv) * c * softmask[i]

                act_size = (softmask * self.n_lvs).sum() * H * W
                return x_bar


class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lvs = 0
        self.bits = 32
        self.act_func = act_func
        self.a = Parameter(torch.Tensor(1))
        self.b = 3/8
        self.c = Parameter(torch.Tensor(1))
        self.d = -3/8

    def initialize(self, n_lvs, offset, diff):
        self.n_lvs = n_lvs
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = x * (F.hardtanh(x + 3, 0, 6) / 6)

        if self.n_lvs == 0:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = x + self.b
            x = F.hardtanh(x / a, 0, 1)
            x = RoundQuant.apply(x, self.n_lvs) * c
            x = x + self.d
            return x 


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lvs = 0
        self.bits = 32
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None
        self.theta = Parameter(torch.Tensor(0))

    def initialize(self, bits):
        if isinstance(bits, list):
            n_lvs = [2**i for i in bits]
        elif isinstance(bits, int) and bits != 32 and bits != 0:
            n_lvs = [2**bits]
        else:
            n_lvs = [self.n_lvs]
        
        if len(n_lvs)>1:
            self.n_lvs = Tensor(n_lvs).view(-1,1,1,1,1)
            self.bits = Tensor(bits)
        else:
            self.n_lvs = n_lvs[0]
            self.bits = bits

        self.theta = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)
        weight = F.hardtanh(self.weight / a, -1, 1)

        if not isinstance(self.n_lvs, torch.Tensor):
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
        else:
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False)
            softmask = softmask.view(-1,1,1,1,1)

            # 1) batched -> out of memory
            weight = weight.repeat(self.n_lvs.shape[0], 1, 1, 1, 1)
            
            try:
                weight = RoundQuant.apply(weight, self.n_lvs) * c
            except Exception as e:
                print(weight.get_device())
                print(self.n_lvs.get_device())
                print(e)
                exit(1)
            weight = softmask * weight
            weight = weight.sum(dim=0)

            '''# 2) for loop
            w_bar = torch.zeros_like(weight)
            for i, n_lv in enumerate(self.n_lvs):
                w_bar += RoundQuant.apply(weight, n_lv) * c * softmask[i,0,0,0,0]'''

        return weight

    def forward(self, x):
        if isinstance(self.n_lvs, int) and self.n_lvs == 0:
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)
        else:
            weight = self._weight_quant()
            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)


class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.n_lvs = 0
        self.bits = 32
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None

    def initialize(self, bits):
        if isinstance(bits, list):
            n_lvs = [2**i for i in bits]
        elif isinstance(bits, int) and bits != 32 and bits != 0:
            n_lvs = [2**bits]
        else:
            n_lvs = [self.n_lvs]
        
        if len(n_lvs)>1:
            self.n_lvs = Tensor(n_lvs).view(-1,1,1)
            self.bits = Tensor(bits)
        else:
            self.n_lvs = n_lvs[0]
            self.bits = bits

        self.theta = Parameter(torch.ones(len(n_lvs))/len(n_lvs))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)

        weight = F.hardtanh(self.weight / a, -1, 1)
        if not isinstance(self.n_lvs, torch.Tensor):
            weight = RoundQuant.apply(weight, self.n_lvs // 2) * c
        else:
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False)
            softmask = softmask.view(-1,1,1)
            weight = weight.repeat(self.n_lvs.shape[0], 1, 1)
            weight = RoundQuant.apply(weight, self.n_lvs) * c
            weight = softmask * weight
            weight = weight.sum(dim=0)

            '''# 2) for loop
            w_bar = torch.zeros_like(weight)
            for i, n_lv in enumerate(self.n_lvs):
                w_bar += RoundQuant.apply(weight, n_lv) * c * softmask[i,0,0]'''

        return weight

    def forward(self, x):
        if isinstance(self.n_lvs, int) and self.n_lvs == 0:
            return F.linear(x, self.weight, self.bias)
        else:
            weight = self._weight_quant()
            return F.linear(x, weight, self.bias)


class Q_Conv2dPad(Q_Conv2d):
    def __init__(self, mode, *args, **kargs):
        super(Q_Conv2dPad, self).__init__(*args, **kargs)
        self.mode = mode

    def forward(self, inputs):
        if self.mode == "HS":
            inputs = F.pad(inputs, self.padding + self.padding, value=-3/8)
        elif self.mode == "RE":
            inputs = F.pad(inputs, self.padding + self.padding, value=0)
        else:
            raise LookupError("Unknown nonlinear")

        if self.n_lvs == 0:
            return F.conv2d(inputs, self.weight, self.bias,
                self.stride, 0, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(inputs, weight, self.bias,
                self.stride, 0, self.dilation, self.groups)



def initialize(model, loader, bits, act=False, weight=False, eps=0.05):
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            if not isinstance(input, torch.Tensor):
                input = input[0]
            input = input.detach().cpu().numpy()

            if isinstance(input, Q_Sym):
                input = np.abs(input)
            elif isinstance(input, Q_HSwish):
                input = input + 3/8

            input = input.reshape(-1)
            input = input[input > 0]
            input = np.sort(input)
            
            if len(input) == 0:
                small, large = 0, 1e-3
            else:
                small, large = input[int(len(input) * eps)], input[int(len(input) * (1-eps))]

            module.initialize(bits, small, large - small)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.initialize(bits)
        
        if isinstance(module, Q_Conv2d):
            O, I, K1, K2 = module.weight.shape
            N, C, H, W = input[0].shape
            s = module.stride[0]
            module.computation = O * I * K1 * K2 * H * W / s / s

        if isinstance(module, Q_Linear):
            O, I = module.weight.shape
            N, I = input[0].shape
            module.computation = O * I

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)


    model.train()
    model.cpu()
    for name, module in model.named_modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish, Q_Conv2d, Q_Linear)):
            if isinstance(module.n_lvs, Tensor):
                module.n_lvs = module.n_lvs.cpu()
            if isinstance(module.bits, Tensor):
                module.bits = module.bits.cpu()


    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break

    model.cuda()
    for name, module in model.named_modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish, Q_Conv2d, Q_Linear)):
            if isinstance(module.n_lvs, Tensor):
                module.n_lvs = module.n_lvs.cuda()
            if isinstance(module.bits, Tensor):
                module.bits = module.bits.cuda()
    for hook in hooks:
        hook.remove()


class Q_Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Q_Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0 
            for module in args:
                if isinstance(module, Q_Sym) or (isinstance(module, Q_HSwish) and idx == 0):
                    self.add_module('-' + str(idx), module)
                else:
                    self.add_module(str(idx), module)
                    idx += 1


class QuantOps(object):
    initialize = initialize
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU
    ReLU6 = Q_ReLU6
    Sym = Q_Sym
    HSwish = Q_HSwish
    Conv2dPad = Q_Conv2dPad        
    Sequential = Q_Sequential
    Linear = Q_Linear