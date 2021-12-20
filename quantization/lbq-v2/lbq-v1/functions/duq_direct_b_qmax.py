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

# https://discuss.pytorch.org/t/torch-round-gradient/28628/6
class Round_fn(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def softmax_init(bits):
    degree = 4
    theta = (bits ** degree)/(bits ** degree).sum
    return theta 
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
        self.bits = Parameter(Tensor([32]))
        self.act_func = act_func
        self.inplace = inplace
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=True)
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def initialize_qonly(self, offset, diff):
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = F.relu(x, self.inplace)
        
        if len(self.bits)==1 and self.bits[0]==32:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)

            nlvs = torch.pow(2, self.bits) # soft forward
            #nlvs = torch.round(bits ** 2) # hard forward
            x = F.hardtanh(x / a, 0, 1)
            x_bar = Round_fn.apply(x.mul(nlvs-1)).div_(nlvs-1) * c 
            #x_bar = RoundQuant.apply(x, nlvs) * c
            return x_bar

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=True)
        self.n_lvs = 2 ** self.bits
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))
        #print("Q_ReLU6")
        #print("self.bits", self.bits)
        #print("self.a", self.a)
        #print("self.c", self.c)


    def initialize_qonly(self, offset, diff):
        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.bits = Parameter(Tensor([32]))
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))

    def initialize(self, bits, offset, diff):
        self.bits = Parameter(Tensor(bits), requires_grad=True)
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def initialize_qonly(self, offset, diff):
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))

    def forward(self, x):
        if len(self.bits)==1 and self.bits[0]==32:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            nlvs = torch.pow(2, self.bits)
            x = F.hardtanh(x / a, -1, 1)
            x_bar = Round_fn.apply(x.mul(nlvs/2-1)).div_(nlvs/2-1) * c 
            #x_bar = RoundQuant.apply(x, torch.round(nlvs / 2)) * c

            return x_bar


################## didn't modify Q_HSwish #################
class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lvs = [1]
        self.bits = [32]
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
            x = Round_fn.apply(x.mul(self.n_lvs-1)).div_(self.n_lvs) * c 
            #x = RoundQuant.apply(x, self.n_lvs) * c
            x = x + self.d
            return x 
##########################################################


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.bits = Parameter(Tensor([32]))
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.weight_old = None
        self.computation = 0


    def initialize(self, bits):
        self.bits = Parameter(Tensor(bits), requires_grad=True)
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))
        
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        #print(self.bits)


    def initialize_qonly(self):
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        #print(self.bits)

        a = F.softplus(self.a)
        c = F.softplus(self.c)
    
        nlvs = torch.pow(2, self.bits)
        w_bar = F.hardtanh(self.weight / a, -1, 1)
        w_bar = Round_fn.apply(w_bar.mul(nlvs/2-1)).div_(nlvs/2-1) * c
        #w_bar = RoundQuant.apply(w_bar, torch.round(nlvs / 2)) * c
        return w_bar

    def forward(self, x):
        if len(self.bits)==1 and self.bits[0]==32:
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)
        else:
            weight = self._weight_quant()
            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)


class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.bits = Parameter(Tensor(1), requires_grad=False)
        self.a = Parameter(Tensor(1))
        self.c = Parameter(Tensor(1))
        self.weight_old = None
        self.computation = 0


    def initialize(self, bits):
        #self.bits = Parameter(Tensor(bits), requires_grad=True)
        self.bits = Parameter(Tensor([8]), requires_grad=False)
        self.a = Parameter(Tensor(len(self.bits)))
        self.c = Parameter(Tensor(len(self.bits)))

        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))
    
    def initialize_qonly(self):
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)
        nlvs = torch.pow(2, self.bits)

        w_bar = F.hardtanh(self.weight / a, -1, 1)
        w_bar = Round_fn.apply(w_bar.mul(nlvs/2-1)).div_(nlvs/2-1) * c
        #w_bar = RoundQuant.apply(w_bar, torch.round(nlvs / 2)) * c
        return w_bar
    

    def forward(self, x):
        if len(self.bits)==1 and self.bits[0]==32:
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

## TODO: boost initialization process for fast test
def initialize(model, loader, bits, act=False, weight=False, eps=0.05):
    if weight: 
        print('==> set up weight bitwidth..')
    elif act: 
        print('==> set up activation bitwidth..')
    if not isinstance(bits, list):
        bits = [bits]
    #if not isinstance(bits, int):
    #    print("Wrong argument for `bits`. ")
    #    raise TypeError

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
    model.cuda()
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


def sample_search_result(model, hard=True):
    print('\n\nsample_search_result\n\n')
    for name, module in model.named_modules():
        if isinstance(module, (Q_Conv2d, Q_Linear, Q_ReLU, Q_Sym, Q_HSwish)):
            for var in ['bits']: #in ['bits', 'n_lvs']:
                before = getattr(module, var)
                setattr(module, var, Parameter(torch.round(getattr(module, var)), requires_grad=False))
                after = getattr(module, var)
                print(f'{before:.4f}->{after:.f4}')
    #---------------
    # TODO: Adopt fracbits' bitwidth selection algorithm 
    #---------------



def extract_bitwidth(model, weight_or_act=None, tau=1):
    print('\nextract_bitwidth\n')
    assert weight_or_act != None
    # i : index of module for printing 
    if weight_or_act == "weight" or weight_or_act == 0:
        i = 1
        module_set = (Q_Conv2d, Q_Linear)
    elif weight_or_act == "act" or weight_or_act == 1:
        i = 2
        module_set = (Q_ReLU, Q_Sym, Q_HSwish)
    else:
        print(f'[ValueError] weight_or_act: {weight_or_act}')
        raise ValueError
    
    list_select = []
    list_prob = []
    str_prob = ''
    for _, m in enumerate(model.modules()):
        if isinstance(m, module_set):
            list_select.append(m.bits.item())
            #prob = F.softmax(m.theta / tau, dim=0)
            #list_select.append(int(m.bits[torch.argmax(prob)].item()))
            #list_prob.append(prob)

            #prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
            #str_prob += f'layer {i} [{", ".join(prob)}]\n'
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
