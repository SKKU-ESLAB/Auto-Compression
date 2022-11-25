import torch
import torch.nn as nn
import torch.nn.functional as F

from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from .group_level_ops import DynamicGroupConv2d, DynamicGroupBatchNorm2d
from utils.config import FLAGS

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# ResNet low channel for basic Block
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inp, outp, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DynamicGroupConv2d(inp, outp, 3, stride, padding=1, bias=False)
        self.bn1 = DynamicGroupBatchNorm2d(outp)
        self.relu1 = nn.ReLU6(inplace=True)
        self.conv2 = DynamicGroupConv2d(outp, outp, 3, stride=1, padding=1, bias=False)
        self.bn2 = DynamicGroupBatchNorm2d(outp)
        self.relu2 = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != outp:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, outp//4, outp//4), 'constant', 0))

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(Model, self).__init__()

        self.width_mult = FLAGS.width_mult_range[-1]
        self.block_setting = [
            #repeat (n)
            2, 2, 2, 2
        ]
        self.channels = make_divisible(64 * self.width_mult)
        self.conv1 = USConv2d(3, self.channels, 1, bias=False,
                              us=[False, True])
        self.bn1 = DynamicGroupBatchNorm2d(self.channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, self.block_setting[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, self.block_setting[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, self.block_setting[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, self.block_setting[3], stride=2)

        self.classifier = USLinear(512, num_classes, us=[True, False])
        
    def _make_layer(self, block, outp, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            outp = make_divisible(outp * self.width_mult)
            layers.append(block(self.channels, outp, stride))
            self.channels = outp * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



