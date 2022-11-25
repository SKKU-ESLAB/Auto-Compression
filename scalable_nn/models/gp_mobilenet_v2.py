import math
import torch.nn as nn

from .group_level_ops import *
from utils.config import FLAGS
#from config import FLAGS

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                DynamicGroupConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,
                    ratio=[1, expand_ratio]),
                DynamicGroupBatchNorm2d(expand_inp, ratio=expand_ratio),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            Conv2d(
                expand_inp, expand_inp, 3, stride, 1, 
                ratio=[expand_ratio, expand_ratio], groups=expand_inp),
            DynamicGroupBatchNorm2d(expand_inp, ratio=expand_ratio),

            nn.ReLU6(inplace=True),

            DynamicGroupConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,
                ratio=[expand_ratio, 1]),
            DynamicGroupBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()
        FLAGS.factor = [112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7]
        print(len(FLAGS.factor))

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #if FLAGS.dataset == 'cifar100':
        #    self.block_setting[2] = [6, 24, 2, 1]

        self.features = []

        width_mult = FLAGS.width_mult
        # head
        assert input_size % 32 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280
        first_stride = 2
        self.features.append(
            nn.Sequential(
                Conv2d(
                    3, channels, 3, 2, 1, bias=False,
                    us=[False, True], ratio=[1, 1]),
                DynamicGroupBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(
                        InvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                DynamicGroupConv2d(
                    channels, self.outp, 1, 1, 0, bias=False),
                DynamicGroupBatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        #if FLAGS.reset_parameters:
        #    self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x
"""
net = Model(100, 32)
net.apply(lambda m: setattr(m, 'width_mult', FLAGS.width_mult))
net.apply(lambda m: setattr(m, 'density', 0.9))

import torch.optim as optim

model_params = []
for params in net.parameters():
    ps = list(params.size())
    if len(ps) == 4 and ps[1] != 1:
        weight_decay = 4e-5
    elif len(ps) == 2:
        weight_decay = 4e-5
    else:
        weight_decay = 0

    item = {'params':params, 'weight_decay':weight_decay,
                'lr': 0.01, 'momentum': 0.9,
                'nesterov':True}
    model_params.append(item)
sgd = optim.SGD(model_params)
lr_scheduler = optim.lr_scheduler.StepLR(sgd, 30)
print(lr_scheduler.last_epoch)
#print(sgd)
"""
