import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .binarization import AlignedGroupedMaskedConv2d, AlignedGroupedMaskedMLP, FilterMaskedConv2d
import torch.nn.functional as F
from collections import OrderedDict
#from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet20', 'resnet32', 'resnet44', 'resnet56']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

###############################################
# CIFAR MODULE
###############################################
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class FilterBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, grouped_rule='l1'):
        super(FilterBasicBlock2, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = FilterMaskedConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv2 = FilterMaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False,
                                                 grouped_rule=grouped_rule)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                FilterMaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=(1,1), stride=stride,padding=0, bias=False, grouped_rule=grouped_rule ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class GroupBasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_shape=(1, 4), grouped_rule='l1'):
        super(GroupBasicBlock2, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = AlignedGroupedMaskedConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.conv2 = AlignedGroupedMaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                AlignedGroupedMaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=(1,1), stride=stride,padding=0, bias=False,group_shape=group_shape, grouped_rule=grouped_rule ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class GroupBottleneck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_shape=(1, 4), grouped_rule='l1'):
        super(GroupBottleneck2, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1 = AlignedGroupedMaskedConv2d(in_planes, planes, kernel_size=(1, 1), stride=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = AlignedGroupedMaskedConv2d(planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1, bias=False)
        self.conv3 = AlignedGroupedMaskedConv2d(planes, self.expansion *planes, kernel_size=(1, 1), stride=1,bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                AlignedGroupedMaskedConv2d(in_planes, self.expansion *planes, kernel_size=(1, 1), bias=False,
                                            stride=stride, group_shape=group_shape, grouped_rule=grouped_rule),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_shape=(1, 4), grouped_rule='l1'):
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 16       
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


###############################################
# CIFAR MODEL
###############################################
def resnet20(num_classes=10, prune_type = 'group', pretrained=True):
    if prune_type is None:
        return CIFAR_ResNet(BasicBlock2, [3, 3, 3], num_classes=num_classes)
    elif prune_type =='group':
        return CIFAR_ResNet(GroupBasicBlock2, [3, 3, 3], num_classes=num_classes)
    elif prune_type =='filter':
        return CIFAR_ResNet(FilterBasicBlock2, [3, 3, 3], num_classes=num_classes)

def resnet32(num_classes=10, prune_type = 'group', pretrained=True):
    if prune_type is None:
        return CIFAR_ResNet(BasicBlock2, [5, 5, 5], num_classes=num_classes)
    elif prune_type =='group':
        return CIFAR_ResNet(GroupBasicBlock2, [5, 5, 5], num_classes=num_classes)
    elif prune_type =='filter':
        return CIFAR_ResNet(FilterBasicBlock2, [5, 5, 5], num_classes=num_classes)

def resnet44(num_classes=10, prune_type = 'group', pretrained=True):
    if prune_type is None:
        return CIFAR_ResNet(BasicBlock2, [7, 7, 7], num_classes=num_classes)
    elif prune_type =='group':
        return CIFAR_ResNet(GroupBasicBlock2, [7, 7, 7], num_classes=num_classes)
    elif prune_type =='filter':
        return CIFAR_ResNet(FilterBasicBlock2, [7, 7, 7], num_classes=num_classes)

def resnet56(num_classes=10, prune_type = 'group', pretrained=True):
    if prune_type is None:
        return CIFAR_ResNet(BasicBlock2, [9, 9, 9], num_classes=num_classes)
    elif prune_type =='group':
        return CIFAR_ResNet(GroupBasicBlock2, [9, 9, 9], num_classes=num_classes)
    elif prune_type =='filter':
        return CIFAR_ResNet(FilterBasicBlock2, [9, 9, 9], num_classes=num_classes)
