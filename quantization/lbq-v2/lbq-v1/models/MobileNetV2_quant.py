from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functions.duq import QuantOps as QuantOps_duq
from functions.hwgq import QuantOps as QuantOps_hwgq
import numpy as np
Q_Sym = None

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, ops, in_planes, out_planes, kernel_size=3, stride=1, groups=1, ReLU6 = True):
        padding = (kernel_size - 1) // 2
        if ReLU6:
            super(ConvBNReLU, self).__init__(
                ops.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),            
                ops.ReLU6(inplace=True)
            )
        else:
            super(ConvBNReLU, self).__init__(
                ops.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes)
            )
            

class InvertedResidual(nn.Module):
    def __init__(self, ops, inp, oup, stride, expand_ratio, first):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if first:
            layers = []
        else:
            layers = [ops.Sym()]

        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(ops, inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(ops, hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            ops.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = ops.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,
                 ops,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()
        global Q_Sym
        if ops == QuantOps_duq:
            print("\n\n** QuantOps_duq is selected **\n\n")
            Q_Sym = QuantOps_duq.Sym
        elif ops == QuantOps_hwgq:
            print("\n\n** QuantOps_hwgq is selected **\n\n")
            Q_Sym = QuantOps_hwgq.Sym

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(ops, 3, input_channel, stride=2)]

        # building inverted residual blocks
        first_block = True
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(ops, input_channel, output_channel, stride, expand_ratio=t, first=first_block))
                input_channel = output_channel
                first_block = False

        # building last several layers
        features.append(ops.Sym())
        features.append(ConvBNReLU(ops, input_channel, self.last_channel, kernel_size=1, ReLU6=False))
        
        # make it nn.Sequential
        self.features = ops.Sequential(*features)

        self.relu6 = ops.ReLU6(False)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            ops.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward(self, x):
        #----------
        # DEBUG: count unique inputs -> result=768(=256*3)
        #print('\n\n\n\n')
        #print(torch.numel(torch.unique(x)))
        #print('\n\n\n\n')
        #exit()
        #----------

        #----------
        # TODO: input quantization to 8-bit (look at fracbits)
        #----------

        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = x.mean([2, 3])
        x = self.relu6(x)
        x = self.classifier(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def mobilenet_v2(ops, pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(ops, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# from class MobileNetV2
    # modified forward: cost passing with indexing
    '''
    def _forward(self, x):
        # first conv
        act_size = 32
        cost = torch.Tensor([0]).cuda()
        
        x, cost = self.features[0][0](x, cost, act_size)
        x = self.features[0][1](x)
        #np.save('0222_lbq_output.npy', x.cpu().detach())
        t = self.features[0][2](x)
        x, act_size = t
        

        # InvertedResidual blocks
        x, cost = self.features[1](x, cost, act_size)
        for i in range(2, 18):
            x, cost = self.features[i](x, cost)
        x, act_size =  self.features[18](x)
        x, cost = self.features[19][0](x, cost, act_size)
        x = self.features[19][1](x)
        
        
        # classifier
        x = F.relu(x, inplace=True)
        x = x.mean([2, 3])
        x, act_size = self.relu6(x)
        x = self.classifier[0](x)
        x, cost = self.classifier[1](x, cost, act_size)
        return x, cost
    '''




# from class InvertedResidual
    '''
    def forward(self, x, cost, act_size=None):
        if self.use_res_connect:
            #print('residual block ----')
            #print(self.conv[0])
            x_identity = x 
            x, act_size = self.conv[0](x)

            x, cost = self.conv[1][0](x, cost, act_size)
            x = self.conv[1][1](x)
            x, act_size = self.conv[1][2](x)

            x, cost = self.conv[2][0](x, cost, act_size)
            x = self.conv[2][1](x)
            x, act_size = self.conv[2][2](x)

            x, cost = self.conv[3](x, cost, act_size)
            x = self.conv[4](x)
            return x_identity + x, cost

        else:
            if isinstance(self.conv[0], Q_Sym):
                try:
                    x, act_size = self.conv[0](x)
                except Exception as e:
                    print('Q_Sym output error')
                    print(type(self.conv[0](x)))
                    print(self.conv[0](x).shape)
                    exit()

                x, cost = self.conv[1][0](x, cost, act_size)
                x = self.conv[1][1](x)
                x, act_size = self.conv[1][2](x)

                x, cost = self.conv[2][0](x, cost, act_size)
                x = self.conv[2][1](x)
                x, act_size = self.conv[2][2](x)

                x, cost = self.conv[3](x, cost, act_size)
                x = self.conv[4](x)
            
            else:
                x, cost = self.conv[0][0](x, cost, act_size)
                x = self.conv[0][1](x)
                x, act_size = self.conv[0][2](x)

                x, cost = self.conv[1](x, cost, act_size)
                x = self.conv[2](x)
            
            # 1. filter ops.Sym 
            #   isinstance(self.conv[0], Q_Sym)
            # 2. process first / second convbnrelu
            #   1) [:1] : conv, input={x, accum_cost, act_size}, output={x, accum_cost}
            #   2) [1:] : bn-relu, input={x}, output={x, act_size}
            # 3. process third conv-bn
            #   1) [:1] : conv, input={x, accum_cost, act_size}, output={x, accum_cost}
            #   2) [1:] : bn, input={x}, output={x}
            return x, cost
    '''

