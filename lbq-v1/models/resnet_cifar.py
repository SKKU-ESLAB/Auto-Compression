'''
ResNet for CIFAR-10/100
This code is modified from the following code:
https://github.com/chenyaofo/CIFAR-pretrained-models/blob/master/cifar_pretrainedmodels/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from functions import *

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', is_qt=False, lq=False, fwlq=False):
        super(BasicBlock, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = LQ_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, 
                is_qt=is_qt, lq=lq, fwlq=fwlq) #block_num=block_num, layer_num=layer_num, index=index, fwlq=fwlq)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = LQ_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, 
                is_qt=is_qt, lq=lq, fwlq=fwlq) #block_num=block_num, layer_num=layer_num, index=index, fwlq=fwlq)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        flops0 = torch.Tensor([0]).cuda()
        if isinstance(x, tuple):
            x, flops0 = x
        out, flops1 = self.conv1(x)  #flops1
        out = F.relu(self.bn1(out))
        out, flops2 = self.conv2(out) #flops2
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        flops0 = flops0 + flops1 + flops2
        return out, flops0 #, flops1+flops2


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, is_qt=False, lq=False, fwlq=False, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, is_qt=is_qt, lq=lq, fwlq=fwlq)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, is_qt=is_qt, lq=lq, fwlq=fwlq)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, is_qt=is_qt, lq=lq, fwlq=fwlq)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, is_qt=False, lq=False, fwlq=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, lq=lq))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("ResNet before conv1: ", type(x))
        
        out = F.relu(self.bn1(self.conv1(x)))
        #print("ResNet after conv1: ", type(out))
        
        out, flops1 = self.layer1(out)  #flops1
        #print('flops1:', flops1)
        #print("ResNet after layer1: ", type(out))
        
        out, flops2 = self.layer2(out)  #flops2
        #print('flops2:', flops2)
        #print("ResNet after layer2: ", type(out))
        
        out, flops3 = self.layer3(out)  #flops3
        #print('flops3:', flops3)
        #print("ResNet after layer3: ", type(out))
        #flops = flops + flops1 + flops2 + flops3
        
        flops = flops1 + flops2 + flops3
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # TODO: linear flops count
        #print(flops)
        #print('[resnet] flops.shape:', flops.shape)
        
        return out, flops


def resnet20(lq=False, is_qt=False, fwlq=False, num_classes=100):
    return ResNet(BasicBlock, [3, 3, 3], is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=num_classes)


def resnet32(lq=False, is_qt=False, fwlq=False, num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=num_classes)


def resnet44(lq=False, is_qt=False, fwlq=False, num_classes=100):
    return ResNet(BasicBlock, [7, 7, 7], is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=num_classes)


def resnet56(lq=False, is_qt=False, fwlq=False, num_classes=100):
    return ResNet(BasicBlock, [9, 9, 9], is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=num_classes)


def resnet110(lq, num_classes):
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202(lq, num_classes):
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
