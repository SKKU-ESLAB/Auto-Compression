import torch
import torch.nn as nn
import torch.nn.functional as F

#from .slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
from .group_level_ops import *
from utils.config import FLAGS
#from config import FLAGS

# ResNet low channel for basic Block
class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inp, outp, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DynamicGroupConv2d(inp, outp, 3, stride=stride, padding=1, bias=False)
        self.bn1 = DynamicGroupBatchNorm2d(outp)
        self.conv2 = DynamicGroupConv2d(outp, outp, 3, stride=1, padding=1, bias=False)
        self.bn2 = DynamicGroupBatchNorm2d(outp)

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != outp:
            self.shortcut = nn.Sequential(
                    DynamicGroupConv2d(inp, outp, 3, stride=stride, padding=1, bais=False),
                    DynamicGroupBatchNorm2d(outp)
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super(Model, self).__init__()
        width_mult = FLAGS.width_mult
        self.conv = nn.Sequential(
                Conv2d(in_channels=3, out_channels=int(64*width_mult), kernel_size=3, stride=1, padding=1, bias=False),
                DynamicGroupBatchNorm2d(int(64*width_mult)),
                nn.ReLU6(),
                DynamicGroupConv2d(in_channels=int(64*width_mult), out_channels=int(128*width_mult), kernel_size=3, stride=1, padding=1, bias=False),
                DynamicGroupBatchNorm2d(int(128*width_mult)),
                nn.ReLU6(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                BasicBlock(inp=int(128*width_mult), outp=int(128*width_mult), stride=1),
                DynamicGroupConv2d(in_channels=int(128*width_mult), out_channels=int(256*width_mult), kernel_size=3, stride=1, padding=1, bias=False),
                DynamicGroupBatchNorm2d(int(256*width_mult)),
                nn.ReLU6(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                DynamicGroupConv2d(in_channels=int(256*width_mult), out_channels=int(256*width_mult), kernel_size=3, stride=1, padding=1, bias=False),
                DynamicGroupBatchNorm2d(int(256*width_mult)),
                nn.ReLU6(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                BasicBlock(inp=int(256*width_mult), outp=int(256*width_mult), stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(int(1024*width_mult), num_classes, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out
"""
import random

torch.autograd.set_detect_anomaly(True)
net = Model(10, 32)
net.apply(lambda m: setattr(m, 'width_mult', FLAGS.width_mult))
net.apply(lambda m: setattr(m, 'density', 1.0))
input = torch.randn(32, 3, 32, 32)
target = torch.tensor([random.randint(0, 9) for _ in range(32)])
output = net(input)
loss = F.cross_entropy(output, target)
loss.backward()
from torchsummary import summary
summary(net, (3, 32, 32), 32, 'cpu')
"""
