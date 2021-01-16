import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def LQ_Conv2d(in_channels, out_channels, kernel_size, **kwargs):
    return lq_conv2d_orig(in_channels, out_channels, kernel_size, **kwargs)

def nan_detect(x):
    if x.isnan().any():
        print(x)
        raise ValueError

class UniformQuantize(InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
            stochastic=False, inplace=False, num_chunks=None, out_half=False, quantize=False,  
            block_num=-1, layer_num=-1, multi=False, index=[], is_act=False):
        if is_act:
            multi=False
        num_chunks = num_chunks = input.shape[0] if num_chunks is None \
                                  else num_chunks
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
        if index != []:
            bit_max = 8
            for i in range(bit_max):
                if len(index[block_num][layer_num][i]) == 0:
                    continue
                else:
                    idx = index[block_num][layer_num][i]
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
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None
 

def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, 
        stochastic=False, inplace=False, quantize=False, 
        block_num=-1, layer_num=-1, multi=False, index=[], is_act=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, stochastic, inplace, num_chunks, False, 
                                   quantize, block_num, layer_num, multi, index, is_act)



class lq_conv2d_orig(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 is_qt=False, tr_gamma=True, lq=False, block_num=-1, layer_num=-1, index=[],
                 fwlq=False):
        super(lq_conv2d_orig, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)
        self.block_num = block_num
        self.layer_num = layer_num
        self.index = index
        self.w_shape = self.weight.shape

        self.is_qt = is_qt
        self.lq = lq
        self.fwlq = fwlq

        #if groups != 1:
        #    self.lq = False

        if lq:
            if fwlq:
                print("filter-wise learning to quantize")
                self.cw = Parameter(torch.ones([out_channels, 1]))
                self.dw = Parameter(torch.ones([out_channels, 1]))
                self.gamma = Parameter(torch.ones([out_channels, 1])) if tr_gamma else 1
            else:
                self.cw = Parameter(torch.Tensor([1]))
                self.dw = Parameter(torch.Tensor([1]))
                self.gamma = Parameter(torch.Tensor([1])) if tr_gamma else 1
            self.cx = Parameter(torch.Tensor([2]))
            self.dx = Parameter(torch.Tensor([2]))
            self.tr_gamma = tr_gamma


    def set_bit_width(self, w_bit, x_bit, initskip):
        self.w_bit = w_bit
        self.x_bit = x_bit
        if isinstance(x_bit, list):
            self.qx = [2**(bit)-1 for bit in x_bit]
            self.theta_x = Parameter(torch.ones([len(x_bit)]/len(x_bit)))
        else: 
            self.qx = 2**(x_bit) - 1
        if isinstance(w_bit, list):
            self.qw = [2**(bit-1)-1 for bit in w_bit]
            self.theta_w = Parameter(torch.ones([len(w_bit)])/len(w_bit))
        else:
            self.qw = 2**(w_bit-1) - 1

        # Read filterwise bitwidth index
        if self.index != []:
            self.qw = torch.ones((self.w_shape[0], 1))
            bit_max = 9
            for i in range(bit_max):
                if len(self.index[self.block_num][self.layer_num][i]) == 0:
                    continue
                else:
                    idx = self.index[self.block_num][self.layer_num][i]
                    self.qw[idx] = 2**(i+1)-1
            self.qw = self.qw.cuda()

        # Initialize c, d
        if self.lq and not initskip:
            with torch.no_grad():
                if self.fwlq:
                    self.cw *= self.weight.abs().mean() #(dim=[1,2,3]).view((-1,1))
                    self.dw *= self.weight.std() #(dim=[1,2,3]).view((-1,1))
                else:
                    self.cw *= self.weight.abs().mean()
                    self.dw *= self.weight.std()

    def bitops_count(self, soft_mask_w=None, soft_mask_x=None):
        x_shape = torch.Tensor([self.x_shape])
        w_shape = torch.Tensor([self.weight.shape])
        flops = x_shape.prod()
        flops *= w_shape.prod()
        # case 1: soft mask w, one value x
        if soft_mask_x == None and soft_mask_w != None:
            bitops = torch.Tensor(self.w_bit).cuda() * soft_mask_w * self.x_bit
            bitops *= flops
            bitops = bitops.sum()
        # case 0: 32-bit w and x 
        elif not (soft_mask_x or soft_mask_w):
            bitops = torch.Tensor([32*32]).cuda()
            bitops *= flops
        return bitops

    def forward(self, input):
        self.x_shape = input.shape[2:]
        soft_mask_w=None
        if self.lq:
            w_abs = self.weight.abs()
            w_sign = self.weight.sign()

            w_abs = w_abs.view(self.w_shape[0], -1)
            w_sign = w_sign.view(self.w_shape[0], -1)

            eps=1e-7
            _dw = self.dw.abs()+eps #.abs()
            _dx = self.dx #s.abs()

            # yejun: d, gamma (no c)
            # Transformer_W
            w_mask1 = (w_abs <= _dw).type(torch.float).detach()
            w_mask2 = (w_abs > _dw).type(torch.float).detach()

            w_cal = w_abs/_dw
            nan_detect(w_cal)
            nan_detect(w_cal.pow(self.gamma))
            w_hat = (w_mask2 * w_sign) + (w_mask1 * (w_cal).pow(self.gamma) * w_sign)
            nan_detect(w_hat)

            # Discretizer_W
            if isinstance(self.qw, list): 
                # 1. learning bitwidth
                w_bar_list = []
                for qw in self.qw:  
                    w_bar = Round.apply(w_hat * qw) / qw
                    nan_detect(w_bar)
                    w_bar_list.append(w_bar)
                soft_mask_w = nn.functional.gumbel_softmax(self.theta_w, tau=1, hard=False)
                w_bar = sum(w * theta for w, theta in zip(w_bar_list, soft_mask_w))
                w_bar = w_bar.view(self.w_shape)
            else:
                # 2. fixed bitwidth
                w_bar = Round.apply(w_hat * self.qw) / self.qw
                nan_detect(w_bar)
                w_bar = w_bar.view(self.w_shape)
                nan_detect(w_bar)

            # Transformer X
            x_mask1 = (input <= self.dx).type(torch.float).detach()
            x_mask2 = (input > self.dx).type(torch.float).detach()
            x_cal = input/self.dx
            nan_detect(x_cal)
            x_hat = x_mask1 * x_cal + x_mask2
            nan_detect(x_hat)

            # Discretizer X
            if isinstance(self.qx, list): 
                # 1. learning bitwidth
                x_bar_list = []
                for qx in self.qx:       
                    x_bar = Round.apply(x_hat * qx) / qx
                    nan_detect(x_bar)
                    x_bar_list.append(x_bar)
                soft_mask_x = nn.functional.gumbel_softmax(self.theta_x, tau=1, hard=False)
                x_bar = sum(x * theta for x, theta in zip(x_bar_list, soft_mask_x))
            else:
                # 2. fixed bitwidth
                x_bar = Round.apply(x_hat * self.qx) / self.qx
                nan_detect(x_bar)
            y = F.conv2d(x_bar, w_bar, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.is_qt:
            if isinstance(self.qw, list):
                w_list = []
                for qw in self.qw:
                    w_list.append(quantize(self.weight, num_bits=qw))
                soft_mask_w = nn.functional.gumbel_softmax(self.theta_w, tau=1, hard=False)
                w = sum(w_ * theta for w_, theta in zip(w_list, soft_mask_w))
                #w= w_bar.view(self.w_shape)

            else:
                w = quantize(self.weight, num_bits=self.w_bit, block_num=self.block_num, layer_num=self.layer_num, multi=True, index=self.index)
            x = quantize(input, num_bits=self.x_bit, is_act=True)
            y = F.conv2d(x, w, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        flops = self.bitops_count(soft_mask_w=soft_mask_w, soft_mask_x=None)
        return y, flops




