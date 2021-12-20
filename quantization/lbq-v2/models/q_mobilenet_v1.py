import math
import numpy as np
import torch.nn as nn
from torch.nn.modules.utils import _pair


from .quantizable_ops import (
    QuantizableConv2d,
    QuantizableLinear
)
from utils.config import FLAGS


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride, input_size):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        l1 = QuantizableConv2d(
                inp, inp, 3, stride, 1, groups=inp, bias=False,
                input_size=input_size)
        l2 = QuantizableConv2d(
                inp, outp, 1, 1, 0, bias=False,
                input_size=l1.output_size)
        layers = [
            l1,
            nn.BatchNorm2d(inp),
            #nn.ReLU6(inplace=True),
            nn.ReLU(inplace=True),

            l2,
            nn.BatchNorm2d(outp),
            #nn.ReLU6(inplace=True),
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)
        self.output_size = l2.output_size

    def forward(self, x):
        return self.body(x)


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        # head
        channels = 32
        first_stride = 2
        l_head = QuantizableConv2d(
            3, channels, 3,
            first_stride, 1, bias=False,
            lamda_w_min=8, lamda_a_min=8,
            weight_only=True,
            input_size=_pair(getattr(FLAGS, 'image_size', (224, 224))))
        self.head = nn.Sequential(
                        l_head,
                        nn.BatchNorm2d(channels),
                        #nn.ReLU6(inplace=True),
                        nn.ReLU(inplace=True),
                    )

        # body
        input_size = l_head.output_size
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c
            for i in range(n):
                if i == 0:
                    layer = DepthwiseSeparableConv(channels, outp, s, input_size)
                else:
                    layer = DepthwiseSeparableConv(channels, outp, 1, input_size)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp
                input_size = layer.output_size

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(0.2),
            QuantizableLinear(
                outp,
                num_classes,
                lamda_w_min=8
            )
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
