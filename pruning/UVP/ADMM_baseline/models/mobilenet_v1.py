import math
import torch.nn as nn
import torch


def make_divisible(v, divisor=4, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        layers = [
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outp),
            nn.ReLU6(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super(MobileNetV1, self).__init__()

        if input_size == 32:
            first_stride = 1
            downsample = 16
        else:
            first_stride = 2
            downsample = 32

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.features = []

        # head
        assert input_size % downsample == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(1024 * width_mult)
        #first_stride = 2
        self.features.append(
            nn.Sequential(
                nn.Conv2d(3, channels, 3, first_stride, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, s))
                else:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, 1))
                channels = outp

        avg_pool_size = input_size // downsample
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.outp, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

