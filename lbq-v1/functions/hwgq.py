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

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336, 5: 0.188, 32: 0}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185, 5: 0.102, 32: 0}


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lvs):    
        return input.mul(n_lvs-1).round_().div_(n_lvs-1)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        #print("type(x):", type(x))
        #print("type(step):", type(step))
        
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Q_ReLU(nn.Module):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.n_lvs = [1]
        self.bits = [32]
        self.act_func = act_func
        self.inplace = inplace
        self.theta = Parameter(Tensor([1]))

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        self.step = []
        self.clip_thr = []
        self.lvls = 2 ** self.bits - 1
        for i, bit in enumerate(self.bits):
            self.step.append(hwgq_steps[bit.item()])
            self.clip_thr.append(self.step[i] * self.lvls[i])
    
    def initialize_qonly(self, offset, diff):
        pass

    def forward(self, x):
        if self.act_func:
            x = F.relu(x, self.inplace)
        
        if len(self.bits)==1 and self.bits[0]==32:
            return x, 32
        else:
            # 1) for loop
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
            x_bar = torch.zeros_like(x)
            for i in range(len(self.bits)):
                y = x.clamp(min=0.0, max=self.clip_thr[i])
                x_bar += _hwgq.apply(y, self.step[i]) * softmask[i]
            act_size = (softmask * self.bits).sum()
            return x_bar, act_size

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        self.theta = Parameter(torch.ones(len(self.n_lvs))/len(self.n_lvs))
        self.step = []
        self.clip_thr = []
        self.lvls = 2 ** self.bits - 1
        for i, bit in enumerate(self.bits):
            self.step.append(hwgq_steps[bit.item()])
            self.clip_thr.append(self.step[i] * self.lvls[i])
        
        for i, val in enumerate(self.clip_thr):
            if val > 6:
                self.clip_thr[i] = 6

    def initialize_qonly(self, offset, diff):
        pass


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.n_lvs = [1]
        self.bits = [32] #Parameter(Tensor([32]), requires_grad=False)
        self.theta = Parameter(Tensor([1]))

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        self.step = []
        self.clip_thr = []
        self.lvls = 2 ** self.bits - 1
        for i, bit in enumerate(self.bits):
            self.step.append(gaussian_steps[bit.item()])
            self.clip_thr.append(self.step[i] * self.lvls[i])

    def initialize_qonly(self, offset, diff):
        pass

    def forward(self, x):
        if len(self.bits)==1 and self.bits[0]==32:
            return x, 32
        else:
            softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
            x_bar = torch.zeros_like(x)
            for i in range(len(self.bits)):
                y = _gauss_quantize.apply(x, self.step[i], self.bits[i])
                x_bar = torch.add(x_bar, y * softmask[i])
            act_size = (softmask * self.bits).sum()
            return x_bar, act_size


################## didn't modify Q_HSwish #################
class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lvs = [1]
        self.bits = [32]

    def initialize(self, n_lvs, offset, diff):
        self.n_lvs = n_lvs
    
    def forward(self, x):
        if self.act_func:
            x = x * (F.hardtanh(x + 3, 0, 6) / 6)

        if len(self.bits)==1 and self.bits[0]==32:
            return x
        else:
            x = F.hardtanh(x, 0, 1)
            x = RoundQuant.apply(x, self.n_lvs) 
            return x 
##########################################################


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lvs = [1]
        self.bits = [32]
        self.weight_old = None
        self.theta = Parameter(Tensor([1]))
        self.computation = 0

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits
        self.steps = []
        for bit in self.bits:
            self.steps.append(gaussian_steps[bit.item()])
        
        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        max_val = self.weight.data.abs().max().item()

    def initialize_qonly(self):
        max_val = self.weight.data.abs().max().item()
        pass

    def _weight_quant(self):
        softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
        w_bar = torch.zeros_like(self.weight)
        weight_std = self.weight.std().item()
        for i, n_lv in enumerate(self.n_lvs):
            quant_weight = _gauss_quantize.apply(self.weight, self.steps[i], self.bits[i])
            w_bar = torch.add(w_bar, quant_weight * softmask[i])
        
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
        self.bits = [32]
        self.weight_old = None
        self.theta = Parameter(Tensor([1]))
        self.computation = 0

    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=False)
        self.n_lvs = 2 ** self.bits

        self.theta = Parameter(torch.ones(len(self.bits))/len(self.bits))
        max_val = self.weight.data.abs().max().item()
        self.steps = []
        for bit in self.bits:
            self.steps.append(gaussian_steps[bit.item()])
    
    def initialize_qonly(self):
        max_val = self.weight.data.abs().max().item()
        pass

    def _weight_quant(self):
        softmask = F.gumbel_softmax(self.theta, tau=1, hard=False, dim=0)
        w_bar = torch.zeros_like(self.weight)
        for i, n_lv in enumerate(self.n_lvs):
            quant_weight = _gauss_quantize.apply(self.weight, self.steps[i], self.bits[i])
            w_bar = torch.add(w_bar, quant_weight * softmask[i])
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
            module.initialize(bits)

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
                for var in ['a', 'c']:
                    setattr(module, var, Parameter(getattr(module, var)[idx].view(1)))
                for var in ['bits', 'n_lvs']:
                    setattr(module, var, Parameter(getattr(module, var)[idx].view(1), requires_grad=False))
                module.theta=Parameter(torch.Tensor([1]), requires_grad=False)
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


def initialize_quantizer(model, loader, eps=0.05):
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
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
            module.initialize_qonly(small, large - small)

        if isinstance(module, (Q_Conv2d, Q_Linear)):
            module.initialize_qonly()
        
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

def transfer_bitwidth(model_src, model_dst): 
    n_lvs_dict={}
    bit_dict={}
    for name, module in model_src.named_modules():
        if isinstance(module, (Q_Conv2d, Q_Linear, Q_ReLU, Q_Sym, Q_HSwish)):
            n_lvs_dict[name] = module.n_lvs.data
            bit_dict[name] = module.bits.data
    for name, module in model_dst.named_modules():
        if isinstance(module, (Q_Conv2d, Q_Linear, Q_ReLU, Q_Sym, Q_HSwish)):
            module.n_lvs.data = n_lvs_dict[name]
            module.bits.data = bit_dict[name]
            print(name)


class QuantOps(object):
    initialize = initialize
    initialize_quantizer = initialize_quantizer
    transfer_bitwidth = transfer_bitwidth
    
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU
    ReLU6 = Q_ReLU6
    Sym = Q_Sym
    HSwish = Q_HSwish
    Conv2dPad = Q_Conv2dPad        
    Sequential = Q_Sequential
    Linear = Q_Linear
