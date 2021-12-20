import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .config import FLAGS


model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
bytes_space = 15
bitops_space = 15
energy_space = 15
latency_space = 15
seconds_space = 15

num_forwards = 10
def compare_models(model_1, model_2, print_id=0):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
      if torch.equal(key_item_1[1], key_item_2[1]):
        pass
      else:
        models_differ += 1
        if (key_item_1[0] == key_item_2[0]):
          print('Rank', print_id,  'Mismtach found at', key_item_1[0])
        else:
          raise Exception
    if models_differ == 0:
      print('Rank {} Models match perfectly! :)'.format(print_id))

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose):
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        #self.n_params = get_params(self)
        self.n_params = ins[1] * outs[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups
        if self.bias is not None:
            self.n_params += outs[1]
        lamda_w = getattr(self, 'lamda_w', None)
        lamda_a = getattr(self, 'lamda_a', None)
        if lamda_w is not None:
            if getattr(FLAGS, 'hard_offset', None):
              lamda_w = torch.round(lamda_w + FLAGS.hard_offset)

            lamda_w = torch.clamp(lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bitw = torch.round(lamda_w)
        else:
            bitw = torch.tensor([32])
        if lamda_a is not None:
            if getattr(FLAGS, 'hard_offset', None):
              lamda_a = torch.round(lamda_a + FLAGS.hard_offset)
            lamda_a = torch.clamp(lamda_a, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bita = torch.round(lamda_a)
        else:
            bita = torch.tensor([32])
        if getattr(self, 'lamda_w_min', None) is not None and lamda_w is not None:
            bitw = torch.clamp(bitw, min=self.lamda_w_min)
        if getattr(self, 'lamda_a_min', None) is not None and lamda_a is not None:
            bita = torch.clamp(bita, min=self.lamda_a_min)
        bitw_max = max(FLAGS.bits_list)
        bita_max = max(FLAGS.bits_list)
        # if getattr(self, 'weight_only', False):
        #     bita = 32
        #     bita_max = 32
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            bitw = bitw.view(self.groups, -1, 1)
        else:
            bitw = bitw.repeat(self.groups, self.out_channels // self.groups, 1)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            bita = bita.view(self.groups, 1, -1)
        else:
            bita = bita.repeat(self.groups, 1, self.in_channels // self.groups)
        self.n_bitops = self.kernel_size[0] * self.kernel_size[1] * \
                        outs[2] * outs[3] * outs[0] * \
                        (bitw * bita).sum() * 1e-9
        self.n_bitops_max = self.n_macs * bitw_max * bita_max * 1e-9
        self.n_bytes = (ins[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups) * bitw.sum() / 8e6
        self.n_bytes_max = (ins[1] * outs[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups) * bitw_max / 8e6
        if self.bias is not None:
            self.n_bytes += outs[1] * 4e-6
            self.n_bytes_max += outs[1] * 4e-6
        self.energy = 0
        self.latency = 0
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        #self.n_params = get_params(self)
        self.n_params = ins[1] * outs[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups
        if self.bias is not None:
            self.n_params += outs[1]
        lamda_w = getattr(self, 'lamda_w', None)
        lamda_a = getattr(self, 'lamda_a', None)
        if lamda_w is not None:
            lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bitw = torch.round(lamda_w)
        else:
            bitw = torch.tensor([32])
        if lamda_a is not None:
            lamda_a = torch.clamp(self.lamda_a, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bita = torch.round(lamda_a)
        else:
            bita = torch.tensor([32])
        if getattr(self, 'lamda_w_min', None) is not None and lamda_w is not None:
            bitw = torch.clamp(bitw, min=self.lamda_w_min)
        if getattr(self, 'lamda_a_min', None) is not None and lamda_a is not None:
            bita = torch.clamp(bita, min=self.lamda_a_min)
        bitw_max = max(FLAGS.bits_list)
        bita_max = max(FLAGS.bits_list)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            bitw = bitw.view(self.groups, -1, 1)
        else:
            bitw = bitw.repeat(self.groups, self.out_channels // self.groups, 1)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            bita = bita.view(self.groups, 1, -1)
        else:
            bita = bita.repeat(self.groups, 1, self.in_channels // self.groups)
        self.n_bitops = self.kernel_size[0] * self.kernel_size[1] * \
                        outs[2] * outs[3] * outs[0] * \
                        (bitw * bita).sum() * 1e-9
        self.n_bitops_max = self.n_macs * bitw_max * bita_max * 1e-9
        self.n_bytes = (ins[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups) * bitw.sum() / 8e6
        self.n_bytes_max = (ins[1] * outs[1] * self.kernel_size[0] * self.kernel_size[1] // self.groups) * bitw_max / 8e6
        if self.bias is not None:
            self.n_bytes += outs[1] * 4e-6
            self.n_bytes_max += outs[1] * 4e-6
        self.energy = 0
        self.latency = 0
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        #self.n_params = get_params(self)
        self.n_params = ins[1] * outs[1]
        if self.bias is not None:
            self.n_params += outs[1]
        lamda_w = getattr(self, 'lamda_w', None)
        lamda_a = getattr(self, 'lamda_a', None)
        if lamda_w is not None:
            lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bitw = torch.round(lamda_w)
        else:
            bitw = torch.tensor([32])
        if lamda_a is not None:
            lamda_a = torch.clamp(self.lamda_a, min(FLAGS.bits_list), max(FLAGS.bits_list))
            bita = torch.round(lamda_a)
        else:
            bita = torch.tensor([32])
        if getattr(self, 'lamda_w_min', None) is not None and lamda_w is not None:
            bitw = torch.clamp(bitw, min=self.lamda_w_min)
        if getattr(self, 'lamda_a_min', None) is not None and lamda_a is not None:
            bita = torch.clamp(bita, min=self.lamda_a_min)
        bitw_max = max(FLAGS.bits_list)
        bita_max = max(FLAGS.bits_list)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            bitw = bitw.view(-1, 1)
        else:
            bitw = bitw.repeat(self.out_features, 1)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            bita = bita.view(1, -1)
        else:
            bita = bita.repeat(1, self.in_features)
        self.n_bitops = (bitw * bita).sum() * 1e-9
        self.n_bitops_max = self.n_macs * bitw_max * bita_max * 1e-9
        self.n_bytes = ins[1] * bitw.sum() / 8e6
        self.n_bytes_max = (ins[1] * outs[1]) * bitw_max / 8e6
        if self.bias is not None:
            self.n_bytes += outs[1] * 4e-6
            self.n_bytes_max += outs[1] * 4e-6
        self.energy = 0
        self.latency = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_bitops = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bitops_max = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bytes = 0
        self.energy = 0
        self.latency = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_bitops = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bitops_max = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bytes = 0
        self.energy = 0
        self.latency = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.BatchNorm2d):
        #self.n_macs = 0
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        #self.n_params = get_params(self)
        self.n_params = 0
        if self.weight is not None:
            self.n_params += 0 #self.num_features
        if self.bias is not None:
            self.n_params += 0 #self.num_features
        self.n_bitops = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bitops_max = 0 #self.n_macs * 32 * 32 * 1e-9
        self.n_bytes = 4e-6 * self.n_params
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_bitops = 0
        self.n_bitops_max = 0
        self.n_bytes = 0
        self.energy = 0
        self.latency = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_bitops += getattr(m, 'n_bitops', 0)
            self.n_bitops_max += getattr(m, 'n_bitops_max', 0)
            self.n_bytes += getattr(m, 'n_bytes', 0)
            self.energy += getattr(m, 'energy', 0)
            self.latency += getattr(m, 'latency', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d, nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and
                t not in ignore_zeros_t):
            print(
                'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:.2f}'.format(self.n_bytes).rjust(bytes_space, ' ') +
            '{:.2f}'.format(self.n_bitops).rjust(bitops_space, ' ') +
            '{:.2f}'.format(self.n_bitops_max).rjust(bitops_space, ' ') +
            '{:.2f}'.format(self.energy).rjust(energy_space, ' ') +
            '{:.2f}'.format(self.latency).rjust(latency_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
      m.register_forward_hook(
        lambda m, input, output: module_profiling(
          m, input, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, batch=1, channel=3, use_cuda=True,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = torch.rand(batch, channel, height, width)
    device = next(model.parameters()).device
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    if verbose is not None:
        print(
            'Item'.ljust(name_space, ' ') +
            'params'.rjust(params_space, ' ') +
            'macs'.rjust(macs_space, ' ') +
            'bytes (MB)'.rjust(bytes_space, ' ') +
            'bitops (B)'.rjust(bitops_space, ' ') +
            'bitops_max (B)'.rjust(bitops_space, ' ') +
            'energy (mJ)'.rjust(energy_space, ' ') +
            'latency (ms)'.rjust(latency_space, ' ') +
            'nanosecs'.rjust(seconds_space, ' '))
    if verbose:
        print(''.center(name_space+params_space+macs_space+bytes_space+bitops_space+bitops_space+energy_space+latency_space+seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(name_space+params_space+macs_space+bytes_space+bitops_space+bitops_space+energy_space+latency_space+seconds_space, '-'))
    if verbose is not None:
        print(
            'Total'.ljust(name_space, ' ') +
            '{:,}'.format(model.n_params).rjust(params_space, ' ') +
            '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
            '{:.2f}'.format(model.n_bytes).rjust(bytes_space, ' ') +
            '{:.5f}'.format(model.n_bitops).rjust(bitops_space, ' ') +
            '{:.5f}'.format(model.n_bitops_max).rjust(bitops_space, ' ') +
            '{:.2f}'.format(model.energy).rjust(energy_space, ' ') +
            '{:.2f}'.format(model.latency).rjust(latency_space, ' ') +
            '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    return model.n_macs, model.n_params, model.n_bitops, model.n_bitops_max, model.n_bytes, model.energy, model.latency
