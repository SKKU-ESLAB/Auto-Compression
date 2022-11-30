import math
import torch.nn as nn


def make_divisible(v, divisor=4, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
                nn.Conv2d(inp, expand_inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            nn.Conv2d(expand_inp, expand_inp, 3, stride, 1, groups=expand_inp, bias=False),
            nn.BatchNorm2d(expand_inp),
            nn.ReLU6(inplace=True),

            nn.Conv2d(expand_inp, outp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outp),
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
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super(Model, self).__init__()

        if input_size == 32:
            first_stride = 1
            second_stride = 1
            downsample = 8
        else:
            first_stride = 2
            second_stride = 2
            downsample = 32

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, second_stride],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #if FLAGS.dataset == 'cifar10':
        #    self.block_setting[2] = [6, 24, 2, 1]

        self.features = []

        # head
        assert input_size % downsample == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280
        #first_stride = 2
        self.features.append(
            nn.Sequential(
                nn.Conv2d(3, channels, 3, first_stride, 1, bias=False),
                nn.BatchNorm2d(channels),
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
                nn.Conv2d(channels, self.outp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // downsample
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x
