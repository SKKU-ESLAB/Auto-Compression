import torch as t

from .quantizer import Quantizer

def soft_pruner(x, block_size, p):
    return x

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def grad_p_scale(c, p):
    x = c
    y = p
    scale = (1. - (p.detach() / c.detach() + 1e-12))
    x_grad = x * scale
    y_grad = p * scale
    return (x - x_grad).detach() + x_grad, (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class weight_quant(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, c, p, thd):
        sign = input.sign()
        distance = c - p + 1e-12
        s = distance / thd
        
        v_g = (input.abs() - p) / s
  
        v_c = t.clamp(v_g, 0, thd) * sign

        v_q = t.round(v_c)

        v_dq = v_q * s
        
        ctx.save_for_backward(input, v_q, c, p, distance)
        ctx.s = s
        ctx.thd = thd
        return v_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        input, v_q, c, p, distance = ctx.saved_tensors
        s = ctx.s
        thd = ctx.thd
        c_mask =(input.abs() > c).float()
        i_mask =(input.abs() <= c).float() * (input.abs() >= p).float()
        s_mask = (v_q == 0.).float()
        sign = input.sign()
        
        v_t = (input.abs() - p) / distance
        
        grad_c = (v_q / thd - v_t * sign) * i_mask
        grad_p = -grad_c - sign * i_mask

        grad_c = grad_c + sign * c_mask
        grad_p = grad_p - sign * c_mask
        
        grad_c = (grad_c * grad_output.clone()).sum()
        grad_p = (grad_p * grad_output.clone()).sum()
        grad_input = grad_output.clone()
        #ste_constant = 2 * s / (2 * p + s)
        
        #grad_p = ((grad_p + s_mask * ste_constant * v_t * (-thd + 0.5) * sign) * grad_output.clone()).sum().reshape(p.shape)
        #grad_c = ((grad_c - s_mask * ste_constant * v_t * sign) * grad_output.clone()).sum().reshape(c.shape)

        #grad_input = ((1. - s_mask) + s_mask * thd * v_t * ste_constant) * grad_output.clone()
        return grad_input, grad_c, grad_p, None

class SLsqQuan(Quantizer):
    def __init__(self, bit, per_channel=False, symmetric = False, all_positive = False, hard_pruning = False, block_size = 4, temperature = 1e-3):
        super().__init__(bit)
        
        self.thd_neg = -2 ** (bit - 1) + 1
        self.thd_pos = 2 ** (bit - 1) - 1
        self.per_channel = per_channel
        self.p= t.nn.Parameter(t.zeros([]))
        self.c = t.nn.Parameter(t.ones(1))
        self.weight_quantizer = weight_quant.apply
        self.soft_mask = None
        self.block_size = block_size
        self.hard_pruning = hard_pruning
        self.temperature = temperature
    
    def calculate_block_sparsity(self,x):
        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw).detach()
        score = (x_reshape.abs().mean(dim = 1, keepdim = True) - self.p).detach()
        hard_mask = (score > 0).float().detach()
        return hard_mask.sum(), hard_mask.numel()

    def soft_pruner(self, x, p):

        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)

        score = x_reshape.abs().mean(dim = 1,keepdim = True) - p
        #score = score / (score.max().detach() * 2)
        temperature = (score.abs().view(-1).sort()[0][int(score.numel()*self.temperature)] * 0.5).detach()
        if not self.hard_pruning:
            _soft_mask = t.nn.functional.sigmoid(score/temperature)
            self.soft_mask = _soft_mask
            self.soft_mask = self.soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
            return self.soft_mask
        
        hard_mask = (score > 0).float()
        hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
        return hard_mask

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            s = x.detach().abs().mean(dim = list(range(1, x.dim())), keepdim = True) * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros_like(self.s))
        else:
            s = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s.clone().detach() * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros([]))
    
    def forward(self, x):
        self.p.data.clamp_(0.,self.c.data)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            #s_grad_scale = (x.abs().max().detach() / (self.thd_pos * x.numel())) ** 0.5
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            #s_grad_scale = (x.abs().max().detach() / (self.thd_pos * x.numel())) ** 0.5
        c_scale = grad_scale(self.c, s_grad_scale)
        p_scale = grad_scale(self.p, s_grad_scale)
        quant_x = self.weight_quantizer(x, c_scale, p_scale, self.thd_pos)
        '''
        sign = x.sign()
        distance = c_scale - p_scale + 1e-12
        s = distance / self.thd_pos
        quant_x = (x.abs() - p_scale) / s
        quant_x = t.clamp(quant_x, 0, self.thd_pos) * sign
        quant_x = (t.round(quant_x) - quant_x).detach() + quant_x
        quant_x = quant_x * s
        '''
        if (len(x.shape) == 4 and x.shape[1] != 1):
            mask = self.soft_pruner(x, p_scale)
            #quant_x = quant_x * mask
            quant_x = (quant_x * mask - quant_x).detach() + quant_x
        return quant_x

class pqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=True, per_channel=False, quant_mode = False, pruning_mode = False, block_size = 4, temperature = 1e-3, hard_pruning = False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones([]))
        self.init_mode = False
        self.quant_mode = quant_mode
        self.pruning_mode = pruning_mode

        self.p= t.nn.Parameter(t.zeros([]))
        self.soft_mask = None
        self.block_size = block_size
        self.hard_pruning = hard_pruning
        self.temperature = temperature

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            self.s = t.nn.Parameter(t.zeros_like(self.s))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.p = t.nn.Parameter(t.zeros_like(self.s))

    def soft_pruner(self, x, p):

        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)

        score = x_reshape.abs().mean(dim = 1,keepdim = True).detach() - p
        score = score / score.abs().max().detach()
        if not self.hard_pruning:
            _soft_mask = t.nn.functional.sigmoid(score/ self.temperature)
            self.soft_mask = _soft_mask
            self.soft_mask = self.soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
            return self.soft_mask
        
        hard_mask = (score > 0).float()
        hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
        return hard_mask

    def forward(self, x):
        self.p.data.clamp_(min = 0.)
        x_r = x
        if self.pruning_mode:
            if (len(x.shape) == 4 and x.shape[1] != 1):
                mask = self.soft_pruner(x, self.p)
                x_r = x_r * mask
        if self.quant_mode:
            if self.init_mode:
                self.init_from(x)
                self.init_mode = False

            if self.per_channel:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)

            s_scale = grad_scale(self.s, s_grad_scale)

            x_r = x_r / s_scale
            x_r = t.clamp(x_r, self.thd_neg, self.thd_pos)
            x_r = round_pass(x_r)
            x_r = x_r * s_scale
        return x_r


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=True, per_channel=False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
            print(self.thd_pos)
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones([]))
        self.init_mode = False

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.init_mode:
            self.init_from(x)
            print(self.s)
            self.init_mode = False
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
'''
class SLsqQuan(Quantizer):
    def __init__(self, bit, per_channel=False, symmetric = False, all_positive = False, hard_pruning = False, block_size = 4, temperature = 1e-3):
        super().__init__(bit)
        
        self.thd_neg = -2 ** (bit - 1) + 1
        self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.p= t.nn.Parameter(t.zeros(1))
        self.c = t.nn.Parameter(t.ones(1))
        self.weight_quantizer = weight_quant.apply
        self.soft_mask = None
        self.block_size = block_size
        self.mask_mean = 0.
        self.hard_pruning = hard_pruning
        self.temperature = temperature

    def soft_pruner(self, x, p):

        co, ci, kh, kw = x.shape
        x_reshape = x.reshape(co // self.block_size, self.block_size, ci, kh, kw)

        score = x_reshape.abs().mean(dim = 1,keepdim = True).detach() - p
        if not self.hard_pruning:
            _soft_mask = t.nn.functional.sigmoid(score/ self.temperature)
            self.soft_mask = _soft_mask
            _soft_mask = _soft_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
            return _soft_mask
        
        hard_mask = (score > 0).float()
        hard_mask = hard_mask.repeat(1, self.block_size, 1, 1, 1).reshape(co,ci,kh,kw)
        return hard_mask

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            s = x.detach().abs().mean(dim = list(range(1, x.dim())), keepdim = True) * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros_like(self.s))
        else:
            s = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            self.c = t.nn.Parameter(s.clone().detach() * self.thd_pos)
            self.p = t.nn.Parameter(t.zeros([]))
    
    def forward(self, x):
        self.p.data.clamp_(0.,self.c.data)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        c_scale = grad_scale(self.c, s_grad_scale)
        p_scale = grad_scale(self.p, s_grad_scale)
        #c_scale = self.c
        #p_scale = self.p
        
        sign = x.sign()
        s = (c_scale - p_scale + 1e-12) / self.thd_pos
        
        quant_x = (x.abs() - p_scale) / s
  
        quant_x = t.clamp(quant_x, 0, self.thd_pos) * sign

        quant_x = (t.round(quant_x) - quant_x).detach() + quant_x

        quant_x = quant_x * s

        if (len(x.shape) == 4):
            mask = self.soft_pruner(x, p_scale)
            quant_x = quant_x * mask
        return quant_x
'''
if __name__ == "__main__":
    module = SLsqQuan(bit = 8)
    x = t.randn((100,4,3,3))
    print(x)
    module.init_from(x = x)
    print(module.c, module.p)
    module.p.data = t.tensor(0.4)
    print(module.c, module.p)
    
    print(x)
    print(module(x))
    
    plt.hist(x.flatten().detach(), bins = 400)
    plt.hist(module(x).flatten().detach(), bins = 400)
    module.hard_pruning = True
    plt.hist(module(x).flatten().detach()+0.1, bins = 400, alpha = 0.4)
    plt.show()
