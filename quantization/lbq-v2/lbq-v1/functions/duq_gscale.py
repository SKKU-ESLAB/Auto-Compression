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

"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
"""
def grad_scale(x, scale):
    yOut = x
    yGrad = x * scale
    return (yOut-yGrad).detach() + yGrad


class Q_ReLU(nn.Module):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.n_lvs = [1]
        self.bits = [32] #Parameter(Tensor([32]), requires_grad=False)
        self.act_func = act_func
        self.inplace = inplace
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.theta = Parameter(Tensor([1]))

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        '''self.bits = bits
        self.n_lvs = [2**i for i in bits]'''
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = F.relu(x, self.inplace)
        
        if len(self.bits)==1 and self.bits[0]==32:
            return x, 32
        else:

            g = 1.0 / torch.sqrt(x.numel() * (self.n_lvs - 1)).to(x.device)
            a = F.softplus(grad_scale(self.a, g))
            c = F.softplus(grad_scale(self.c, g))

            # 1) for loop
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
            softmask = softmask
            x_bar = torch.zeros_like(x)
            for i, n_lv in enumerate(self.n_lvs):
                x_temp = F.hardtanh(x / a[i], 0, 1)
                #x_bar += RoundQuant.apply(x, n_lv) * c * softmask[i]
                x_bar = torch.add(x_bar, RoundQuant.apply(x_temp, n_lv) * c[i] * softmask[i])
            act_size = (softmask * self.bits).sum()
            return x_bar, act_size

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        '''self.bits = bits
        self.n_lvs = [2**i for i in bits]'''
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))
        self.theta = Parameter(torch.ones(len(self.n_lvs))/len(self.n_lvs))

        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.n_lvs = [1]
        self.bits = [32] #Parameter(Tensor([32]), requires_grad=False)
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.theta = Parameter(Tensor([1]))

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        #self.bits = bits
        #self.n_lvs = [2**i for i in bits]
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))

    def forward(self, x):
        if len(self.bits)==1 and self.bits[0]==32:
            return x, 32
        else:
            g = 1.0 / torch.sqrt(x.numel() * (self.n_lvs // 2 - 1)).to(x.device)
            a = F.softplus(grad_scale(self.a, g))
            c = F.softplus(grad_scale(self.c, g))
            
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
            softmask = softmask
            x_bar = torch.zeros_like(x)
            for i, n_lv in enumerate(self.n_lvs):
                x_temp = F.hardtanh(x / a[i], -1, 1)
                x_bar = torch.add(x_bar, RoundQuant.apply(x_temp, n_lv // 2) * c[i] * softmask[i])
            act_size = (softmask * self.bits).sum()
            return x_bar, act_size

################## didn't modify Q_HSwish #################
class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lvs = [1]
        self.bits = [32]#Parameter(Tensor([32]), requires_grad=False)
        self.act_func = act_func
        self.a = Parameter(Tensor(1))
        self.b = 3/8
        self.c = Parameter(Tensor(1))
        self.d = -3/8

    def initialize(self, n_lvs, offset, diff):
        self.n_lvs = n_lvs
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = x * (F.hardtanh(x + 3, 0, 6) / 6)

        if len(self.bits)==1 and self.bits[0]==32:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = x + self.b
            x = F.hardtanh(x / a, 0, 1)
            x = RoundQuant.apply(x, self.n_lvs) * c
            x = x + self.d
            return x 
##########################################################

class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lvs = [1]
        self.bits = [32] #Parameter(Tensor([32]), requires_grad=False)
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.weight_old = None
        self.theta = Parameter(Tensor([1]))
        self.computation = 0

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        
        '''self.bits = bits
        #self.n_lvs = [2**i for i in bits]'''
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))
        
        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        g = 1.0 / torch.sqrt(self.weight.numel() * (self.n_lvs // 2 - 1)).to(self.weight.device)
        a = F.softplus(grad_scale(self.a, g))
        c = F.softplus(grad_scale(self.c, g))
        
        softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
        w_bar = torch.zeros_like(self.weight)
        for i, n_lv in enumerate(self.n_lvs):
            weight = F.hardtanh(self.weight / a[i], -1, 1)
            w_bar = torch.add(w_bar, RoundQuant.apply(weight, n_lv // 2) * c[i] * softmask[i])
        bitwidth = (softmask * self.bits).sum()

        return w_bar, bitwidth

    def forward(self, x, cost, act_size=None):
        if len(self.bits)==1 and self.bits[0]==32:
            cost += act_size * 32 * self.computation
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups), cost
        else:
            weight, bitwidth = self._weight_quant()
            cost += act_size * bitwidth * self.computation
            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups), cost


class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.n_lvs = [0]
        self.bits = [32]#Parameter(Tensor([32]), requires_grad=False)
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.weight_old = None
        self.theta = Parameter(Tensor([1]))
        self.computation = 0

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        '''self.bits = bits
        self.n_lvs = [2**i for i in bits]'''
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        g = 1.0 / torch.sqrt(self.weight.numel() * (self.n_lvs // 2 - 1)).to(self.weight.device)
        a = F.softplus(grad_scale(self.a, g))
        c = F.softplus(grad_scale(self.c, g))

        softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
        w_bar = torch.zeros_like(self.weight)
        for i, n_lv in enumerate(self.n_lvs):
            weight = F.hardtanh(self.weight / a[i], -1, 1)                
            w_bar = torch.add(w_bar, RoundQuant.apply(weight, n_lv // 2) * c[i] * softmask[i])
        bitwidth = (softmask * self.bits).sum()
        return w_bar, bitwidth

    

    def forward(self, x, cost, act_size=None):
        if len(self.bits)==1 and self.bits[0]==32:
            cost += act_size * 32 * self.computation
            return F.linear(x, self.weight, self.bias), cost
        else:
            weight, bitwidth = self._weight_quant()
            cost += act_size * bitwidth * self.computation
            return F.linear(x, weight, self.bias), cost


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
    if isinstance(bits, int):
        bits = [bits]
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            if not isinstance(input, list):
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
        
        if isinstance(module, Q_Conv2d) and weight:
            O, I, K1, K2 = module.weight.shape
            N, C, H, W = input[0].shape
            s = module.stride[0]
            module.computation = O * I * K1 * K2 * H * W / s / s

        if isinstance(module, Q_Linear) and weight:
            O, I = module.weight.shape
            N, I = input[0].shape
            module.computation = O * I

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    model.train()

    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input.cuda())
            else:
                output = model(input.cuda())
        break

    model.cuda()
    for hook in hooks:
        hook.remove()


def sample_search_result(model, hard=True, print=True):
    if hard:
        for name, module in model.named_modules():
            if isinstance(module, (Q_Conv2d, Q_Linear, Q_ReLU, Q_Sym, Q_HSwish)):
                idx = torch.argmax(module.theta)
                for var in ['a', 'c', 'bits', 'n_lvs']:
                    setattr(module, var, Parameter(getattr(module, var)[idx].view(1)))
                module.theta=Parameter(torch.Tensor([1]))
    else: 
        # TODO: stochastic sampling
        raise NotImplementedError


def extract_bitwidth(model, weight_or_act=None):
    if weight_or_act == "weight":
        i = 1
        module_set = (Q_Conv2d, Q_Linear)
    elif weight_or_act == "act":
        i = 2
        module_set = (Q_ReLU, Q_Sym, Q_HSwish)
    
    list_select = []
    list_prob = []
    str_prob = ''
    for _, m in enumerate(model.modules()):
        if isinstance(m, module_set):
            prob = F.softmax(m.theta)
            list_select.append(int(m.bits[torch.argmax(prob)].item()))
            list_prob.append(prob)

            prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
            str_prob += f'layer {i} [{", ".join(prob)}]\n'
        i += 1
    str_select = f'{weight_or_act} bitwidth select: \n' + ", ".join(map(str, list_select))
    return list_select, list_prob, str_select, str_prob


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
