# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MyModule(nn.Module):
    '''
    def forward(self, x):
        raise NotImplementedError
    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError
    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
    '''
class MyNetwork(nn.Module):
    '''
    def forward(self, x):
        raise NotImplementedError
    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError
    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError
    """ implemented methods """
    '''
    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init):
        """ Conv2d, BatchNorm2d, BatchNorm1d, Linear, """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_parameters(self, keys=None, mode='include', exclude_set=None):
        if exclude_set is None:
            exclude_set = {}
        if keys is None:
            for name, param in self.named_parameters():
                if name not in exclude_set:
                    yield param
        elif mode == 'include':
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and name not in exclude_set:
                    yield param
        elif mode == 'exclude':
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and name not in exclude_set:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self, exclude_set=None):
        return self.get_parameters(exclude_set=exclude_set)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        #return Hswish(inplace=inplace)
        return nn.Hardswish(inplace = inplace)
        #return nn.ReLU(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return nn.HardSigmoid(inplace=inplace)
        #return Hsigmoid(inplace=inplace)
        #return nn.ReLU(inplace=inplace)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class ShuffleLayer(nn.Module):

    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION
        num_mid = make_divisible(self.channel // self.reduction, divisor=8)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            #('h_sigmoid', Hsigmoid(inplace=True)),
            ('h_sigmoid', nn.Hardsigmoid(inplace=True)),
            #('relu', nn.ReLU(inplace=True)),
        ]))
        self.skip_mul = nn.quantized.FloatFunctional()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return self.skip_mul.mul(x, y)


def profile(model, input_size, custom_ops=None):

    def count_convNd(m, _, y):
        cin = m.in_channels

        kernel_ops = m.weight.size()[2] * m.weight.size()[3]
        ops_per_element = kernel_ops
        output_elements = y.nelement()

        # cout x oW x oH
        total_ops = cin * output_elements * ops_per_element // m.groups
        m.total_ops = torch.zeros(1).fill_(total_ops)


    def count_linear(m, _, __):
        total_ops = m.in_features * m.out_features

        m.total_ops = torch.zeros(1).fill_(total_ops)


    register_hooks = {
        nn.Conv1d: count_convNd,
        nn.Conv2d: count_convNd,
        nn.Conv3d: count_convNd,
        ######################################
        nn.Linear: count_linear,
    }

    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.zeros(1).fill_(p.numel())

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params
