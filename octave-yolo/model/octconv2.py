import torch
import torch.nn as nn
import math


class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.3333, alpha_out=0.3333, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv2, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.doubledownsample = nn.AvgPool2d(kernel_size=(4, 4), stride=4)
        self.doubleupsample = nn.Upsample(scale_factor=4, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        
        self.conv_l2l = None if alpha_out == 0 else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2m = None if alpha_out == 0 else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2h = None if alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), out_channels - 2*(math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)
        
        self.conv_m2l = None if alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_m2m = None if alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_m2h = None if alpha_out == 1 else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), out_channels - 2*(math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)
        
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2m = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), (math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(math.ceil(alpha_in * in_channels), out_channels - 2*(math.ceil(alpha_out * out_channels)),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)
        

    def forward(self, x):
        x_h, x_m, x_l = x if type(x) is tuple else (x, None, None)

        x_h2h = self.conv_h2h(x_h)
        x_h2m = self.conv_h2m(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        x_h2l = self.conv_h2l(self.doubledownsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_m2h = self.upsample(self.conv_m2h(x_m))
            x_m2m = self.conv_m2m(x_m) if self.alpha_out > 0 else None
            x_m2l = self.conv_m2l(self.downsample(x_m)) if self.alpha_out > 0 else None

            x_l2h = self.doubleupsample(self.conv_l2h(x_l))
            x_l2m = self.upsample(self.conv_l2m(x_l)) if self.alpha_out > 0 else None
            x_l2l = self.conv_l2l(x_l) if self.alpha_out > 0 else None

            x_h = x_l2h + x_m2h + x_h2h
            x_m = x_l2m + x_m2m + x_h2m if x_l2m is not None and x_m2m is not None and x_h2m is not None else None
            x_l = x_l2l + x_m2l + x_h2l if x_l2l is not None and x_m2l is not None and x_h2l is not None else None
            return x_h, x_m, x_l
        else:
            return x_h2h, x_h2m, x_h2l


class Conv_BN2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.3333, alpha_out=0.3333, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv2(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(math.ceil(out_channels * (1 - alpha_out)))
        self.bn_m = None if alpha_out == 0 else norm_layer(math.ceil(out_channels * alpha_out))
        self.bn_l = None if alpha_out == 0 else norm_layer(math.ceil(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_m, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_m(x_m) if x_m is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_m, x_l


class Conv_BN_ACT2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.3333, alpha_out=0.3333, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv2(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(math.ceil(out_channels * (1 - alpha_out)))
        self.bn_m = None if alpha_out == 0 else norm_layer(math.ceil(out_channels * alpha_out))
        self.bn_l = None if alpha_out == 0 else norm_layer(math.ceil(out_channels * alpha_out))
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_m, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_m(x_m) if x_m is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_m, x_l
