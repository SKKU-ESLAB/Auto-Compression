import math
import torch.nn as nn

from .group_level_ops import *
from typing import Union, List, Dict, Any, cast

class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()

        self.block_setting = [
            64, 'M', 
            128, 128, 'M'. 
            256, 256, 256, 'M', 
            512, 512, 512, 'M', 
            512, 512, 512, 'M'
            ]

        self.features = [
            Conv2d(3, 64, 3, 1, 1, bias=False, us=[False, False], ratio=[1, 1]),
            DynamicGroupBatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        self.features.extend(self.make_layers(self.block_setting))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
            )
    def forward(self, input):
        out = self.features(input)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def make_layers(cfg: List[str, int]]) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 64

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                layers += [
                    DynamicGroupConv2d(in_channels, v, 3, 1, 1, bias=True,
                        us=[False, False],ratio = [1, 1]),
                    DynamicGroupBatchNorm2d(v),
                    nn.ReLU(inplace=True)
                    ]
                in_channels = v
        return layers



