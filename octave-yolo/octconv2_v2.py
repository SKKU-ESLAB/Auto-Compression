import torch
import torch.nn as nn
import math
import pdb

class OctaveConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_l_in, alpha_m_in, alpha_h_in, alpha_l_out, alpha_m_out, alpha_h_out, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv2, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.doubledownsample = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        self.doubleupsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.pad = torch.nn.ZeroPad2d((0, 1, 0, 1))
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        self.alpha_l_in, self.alpha_l_out = alpha_l_in, alpha_l_out
        self.alpha_m_in, self.alpha_m_out = alpha_m_in, alpha_m_out
        self.alpha_h_in, self.alpha_h_out = alpha_h_in, alpha_h_out


        self.conv_l2l = nn.Conv2d((alpha_l_in * in_channels), (alpha_l_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2m = nn.Conv2d((alpha_l_in * in_channels), (alpha_m_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2h = nn.Conv2d((alpha_l_in * in_channels), (alpha_h_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        
        self.conv_m2l = nn.Conv2d((alpha_m_in * in_channels), (alpha_l_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_m2m = nn.Conv2d((alpha_m_in * in_channels), (alpha_m_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_m2h = nn.Conv2d((alpha_m_in * in_channels), (alpha_h_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        
        self.conv_h2l = nn.Conv2d((alpha_h_in * in_channels), (alpha_l_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2m = nn.Conv2d((alpha_h_in * in_channels), (alpha_m_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = nn.Conv2d((alpha_h_in * in_channels), (alpha_h_out * out_channels),
                                kernel_size, 1, padding, dilation, groups, bias)
        

    def forward(self, x):
        x_h, x_m, x_l = x if type(x) is tuple else (x, None, None)
        x_h2h = self.conv_h2h(x_h)
        x_h2m = self.conv_h2m(self.downsample(x_h))
        x_h2l = self.conv_h2l(self.doubledownsample(x_h))
        if x_l is not None:
            x_m2h = self.upsample(self.conv_m2h(x_m))
            #if (x_m.shape[3]%2==1):
            #    x_m2h = self.pad(x_m2h)
            x_m2m = self.conv_m2m(x_m)
            x_m2l = self.conv_m2l(self.downsample(x_m))

            x_l2h = self.doubleupsample(self.conv_l2h(x_l))
            #if (x_h.shape[3]%2==1):
            #    x_l2h = self.pad(x_l2h)
            x_l2m = self.upsample(self.conv_l2m(x_l))
            #if (x_h.shape[3]%2==1):
            #    x_l2m = self.pad(x_l2m)
            x_l2l = self.conv_l2l(x_l)
            x_h = x_l2h + x_m2h + x_h2h
            
            x_m = x_l2m + x_m2m + x_h2m
            x_l = x_l2l + x_m2l + x_h2l
            return x_h, x_m, x_l
        else:
            return x_h2h, x_h2m, x_h2l


class Conv_BN2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_l_in, alpha_m_in, alpha_h_in, alpha_l_out, alpha_m_out, alpha_h_out, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN2, self).__init__()
        self.conv = OctaveConv2(in_channels, out_channels, kernel_size, alpha_l_in, alpha_m_in, alpha_h_in, alpha_l_out, alpha_m_out, alpha_h_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer((alpha_h_out * out_channels))
        self.bn_m = norm_layer((alpha_m_out * out_channels))
        self.bn_l = norm_layer((alpha_l_out * out_channels))

    def forward(self, x):
        x_h, x_m, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_m = self.bn_m(x_m) if x_m is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_m, x_l


class Conv_BN_ACT2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_l_in, alpha_m_in, alpha_h_in, alpha_l_out, alpha_m_out, alpha_h_out, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT2, self).__init__()
        self.conv = OctaveConv2(in_channels, out_channels, kernel_size, alpha_l_in, alpha_m_in, alpha_h_in, alpha_l_out, alpha_m_out, alpha_h_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer((alpha_h_out * out_channels))
        self.bn_m = norm_layer((alpha_m_out * out_channels))
        self.bn_l = norm_layer((alpha_l_out * out_channels))
        self.act = activation_layer

    def forward(self, x):
        #print(x[0].shape,x[1].shape,x[2].shape)
        x_h, x_m, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_m = self.bn_m(x_m)
        x_l = self.bn_l(x_l)
        #print(x_h.shape,x_m.shape,x_l.shape)
        return x_h, x_m, x_l
