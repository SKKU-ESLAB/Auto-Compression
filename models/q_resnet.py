import math
import numpy as np
import torch.nn as nn
from torch.nn.modules.utils import _pair


from .quantizable_ops import (
    QuantizableConv2d,
    QuantizableLinear,
    MaxPool2d
)
from utils.config import FLAGS


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, outp, stride, input_size):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        l1 = QuantizableConv2d(inp, outp, 3, stride, 1, bias=False, input_size=input_size)
        l2 = QuantizableConv2d(outp, outp, 3, 1, 1, bias=False, input_size=l1.output_size)
        layers = [
            l1,
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),

            l2,
            nn.BatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)
        self.output_size = l2.output_size

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                QuantizableConv2d(inp, outp, 1, stride=stride, bias=False, input_size=input_size),
                nn.BatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inp, outp, stride, input_size):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]

        midp = outp//4
        l1 = QuantizableConv2d(inp, midp, 1, 1, 0, bias=False, input_size=input_size)
        l2 = QuantizableConv2d(midp, midp, 3, stride, 1, bias=False, input_size=l1.output_size)
        l3 = QuantizableConv2d(midp, outp, 1, 1, 0, bias=False, input_size=l2.output_size)
        layers = [
            l1,
            nn.BatchNorm2d(midp),
            nn.ReLU(inplace=True),

            l2,
            nn.BatchNorm2d(midp),
            nn.ReLU(inplace=True),

            l3,
            nn.BatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)
        self.output_size = l3.output_size

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                QuantizableConv2d(inp, outp, 1, stride=stride, bias=False, input_size=input_size),
                nn.BatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        if FLAGS.depth in [18, 34]:
            block = BasicBlock
        elif FLAGS.depth in [50, 101, 152]:
            block = Bottleneck

        # head
        channels = 64
        if getattr(FLAGS, 'channel_half', False):
            channels = 32
        
        l_head = QuantizableConv2d(
            3, channels, 7, 2, 3,
            bias=False,
            lamda_w_min=8, lamda_a_min=8,
            weight_only=True,
            input_size=_pair(getattr(FLAGS, 'image_size', (224, 224))))
        mp_head = MaxPool2d(3, 2, 1, input_size=l_head.output_size)
        self.head = nn.Sequential(
                        l_head,
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                        mp_head,
                    )

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]

        feats = [64, 128, 256, 512]
        if getattr(FLAGS, 'channel_half', False):
            feats = [32, 64, 128, 256]

        # body
        input_size = mp_head.output_size
        for idx, n in enumerate(self.block_setting):
            outp = feats[idx]*block.expansion
            for i in range(n):
                if i == 0 and idx != 0:
                    layer = block(channels, outp, 2, input_size)
                else:
                    layer = block(channels, outp, 1, input_size)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp
                input_size = layer.output_size

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
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
        for idx, n in enumerate(self.block_setting):
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
