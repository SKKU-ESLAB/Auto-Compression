import math
import torch
import torch.nn as nn

"""
Function for activation binarization
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


class AlignedGroupedMaskedMLP(nn.Module):
    def __init__(self, in_size, out_size, group_shape=(1, 4), grouped_rule='l1'):
        super(AlignedGroupedMaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.step = BinaryStep.apply

        self.group_shape = group_shape
        self.grouped_rule = grouped_rule
        self.threshold = nn.Parameter(torch.Tensor(out_size//group_shape[0]))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0)

    def forward(self, input):
        weight_shape = self.weight.shape
        threshold_shape = self.threshold.shape
        threshold = self.threshold.view(threshold_shape[0], -1)
        weight = self.weight.view(threshold_shape[0], self.group_shape[0], -1, self.group_shape[1])
        if self.grouped_rule == 'l1':
            grouped_weight = weight.abs().sum(dim=(1, 3))
        elif self.grouped_rule == 'l2':
            grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
        grouped_weight = grouped_weight - threshold
        grouped_mask = self.step(grouped_weight)
        mask = grouped_mask.repeat_interleave(self.group_shape[0], dim=0).repeat_interleave(self.group_shape[1], dim=1)
        ratio = torch.sum(grouped_mask) / grouped_mask.numel()
        #print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0)
            threshold = self.threshold.view(threshold_shape[0], -1)
            weight = self.weight.view(threshold_shape[0], self.group_shape[0], -1, self.group_shape[1])
            if self.grouped_rule == 'l1':
                grouped_weight = weight.abs().sum(dim=(1, 3))
            elif self.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
            grouped_weight = grouped_weight - threshold
            grouped_mask = self.step(grouped_weight)
            mask = grouped_mask.repeat_interleave(self.group_shape[0], dim=0).repeat_interleave(self.group_shape[1], dim=1)
        masked_weight = self.weight * mask
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output




class AlignedGroupedMaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, group_shape=(1, 4), grouped_rule='l1'):
        super(AlignedGroupedMaskedConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ## define weight
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.step = BinaryStep.apply

        self.group_shape = group_shape
        self.grouped_rule = grouped_rule
        self.threshold = nn.Parameter(torch.Tensor(out_c//group_shape[0]))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        weight_shape = self.weight.shape
        threshold_shape = self.threshold.shape
        threshold = self.threshold.view(threshold_shape[0], -1)
        weight = self.weight.view(threshold_shape[0], self.group_shape[0], -1, self.group_shape[1])
        if self.grouped_rule == 'l1':
            grouped_weight = weight.abs().sum(dim=(1, 3))
        elif self.grouped_rule == 'l2':
            grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
        grouped_weight = grouped_weight - threshold
        grouped_mask = self.step(grouped_weight)
        mask = grouped_mask.repeat_interleave(self.group_shape[0], dim=0).repeat_interleave(self.group_shape[1], dim=1)
        mask = mask.view(weight_shape)
        ratio = torch.sum(grouped_mask) / grouped_mask.numel()
        # print("threshold {:3f}".format(self.threshold[0]))
        # print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            threshold = self.threshold.view(threshold_shape[0], -1)
            weight = self.weight.view(threshold_shape[0], self.group_shape[0], -1, self.group_shape[1])
            if self.grouped_rule == 'l1':
                grouped_weight = weight.abs().sum(dim=(1, 3))
            elif self.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
            grouped_weight = grouped_weight - threshold
            grouped_mask = self.step(grouped_weight)
            mask = grouped_mask.repeat_interleave(self.group_shape[0], dim=0).repeat_interleave(self.group_shape[1], dim=1)
            mask = mask.view(weight_shape)
        masked_weight = self.weight * mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out

class FilterMaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, grouped_rule='l1'):
        super(FilterMaskedConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ## define weight
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.step = BinaryStep.apply

        self.grouped_rule = grouped_rule
        self.threshold = nn.Parameter(torch.Tensor(1))#out_c
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.01)

    def forward(self, x):
        weight_shape = self.weight.shape
        threshold_shape = self.threshold.shape
        threshold = self.threshold.view(threshold_shape[0])#out
        weight = self.weight.view(weight_shape[0], -1) # out_c,int_c
        #grouped_weight = weight.abs()
        if self.grouped_rule == 'l1':
            grouped_weight = weight.abs().mean(dim=(1)) # out_c
        elif self.grouped_rule == 'l2':
            grouped_weight = weight.pow(2).mean(dim=(1)).sqrt()
        grouped_weight = grouped_weight - threshold # out_c - out_c
        grouped_mask = self.step(grouped_weight)
        grouped_mask = grouped_mask.view(weight_shape[0],-1).repeat_interleave(weight.shape[1], dim=1)
        mask = grouped_mask.view(weight_shape)
        ratio = torch.sum(grouped_mask) / grouped_mask.numel()
        # print("threshold {:3f}".format(self.threshold[0]))
        # print("keep ratio {:.2f}".format(ratio))
        """
        if ratio <= 0.000001:
            print('low ratio')
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            threshold = self.threshold.view(threshold_shape[0])
            weight = self.weight.view(weight_shape[0], -1)
            if self.grouped_rule == 'l1':
                grouped_weight = weight.abs().mean(dim=(1))
            elif self.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).mean(dim=(1)).sqrt()
            grouped_weight = grouped_weight - threshold
            grouped_mask = self.step(grouped_weight)
            grouped_mask = grouped_mask.view(weight_shape[0],-1).repeat_interleave(weight.shape[1], dim=1)
            mask = grouped_mask.view(weight_shape)
        """

        masked_weight = self.weight * mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out        