import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .binarization import AlignedGroupedMaskedConv2d, AlignedGroupedMaskedMLP, FilterMaskedConv2d
import torch.nn.functional as F
from collections import OrderedDict
#from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


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
# IMAGENET MODULE
###############################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Aligned group level pruning
class GroupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group_shape=(1,4), grouped_rule='l1'):
        super(GroupBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = AlignedGroupedMaskedConv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = AlignedGroupedMaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class GroupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group_shape=(1,4), grouped_rule='l1'):
        super(GroupBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        #self.conv1 = conv1x1(inplanes, width)
        self.conv1 = AlignedGroupedMaskedConv2d(inplanes, width, kernel_size=(1, 1), stride=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn1 = norm_layer(width)
        #self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv2 = AlignedGroupedMaskedConv2d(width, width, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn2 = norm_layer(width)
        #self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = AlignedGroupedMaskedConv2d(width, planes * self.expansion, kernel_size=(1, 1), stride=1, bias=False,
                                                group_shape=group_shape, grouped_rule=grouped_rule)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Aligned filter level pruning
class FilterBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, grouped_rule='l1'):
        super(FilterBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = FilterMaskedConv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = FilterMaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class FilterBottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, grouped_rule='l1'):
        super(FilterBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        #self.conv1 = conv1x1(inplanes, width)
        self.conv1 = FilterMaskedConv2d(inplanes, width, kernel_size=(1, 1), stride=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn1 = norm_layer(width)
        #self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv2 = FilterMaskedConv2d(width, width, kernel_size=(3, 3), stride=stride, padding=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn2 = norm_layer(width)
        #self.conv3 = conv1x1(width, planes * self.expansion)
        self.conv3 = FilterMaskedConv2d(width, planes * self.expansion, kernel_size=(1, 1), stride=1, bias=False,
                                                grouped_rule=grouped_rule)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, prune_type = None): # prune_type None--> not prune
        super(ResNet, self).__init__()
        self.prune_type = prune_type
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.prune_type is None:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            elif self.prune_type == 'group':
                downsample = nn.Sequential(
                    AlignedGroupedMaskedConv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), bias=False,
                                            stride=stride, group_shape=(1,4), grouped_rule='l1'),
                    norm_layer(planes * block.expansion),
                )

            elif self.prune_type == 'filter':
                downsample = nn.Sequential(
                    FilterMaskedConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), stride=stride, bias=False, grouped_rule='l1'),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, prune_type, **kwargs):
    model = ResNet(block, layers, prune_type=prune_type, **kwargs)
    if pretrained:
        if prune_type is None:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
            model.load_state_dict(state_dict)
        elif (prune_type=='group') or (prune_type=='filter'):
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
            #
            temp_dict = OrderedDict()
            for name,parms in net.named_parameters():
                temp_dict[name] = parms
            
            for name in state_dict:
                temp_dict[name] = state_dict[name]

            model.load_state_dict(temp_dict)
    return model

###############################################
# IMAGENET MODEL
###############################################
def resnet18(pretrained=False, progress=True, prune_type = None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        prune_type (str) : If None, not prune / choice filter or group
    """
    if prune_type is None:
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, prune_type,
                    **kwargs)
    elif prune_type =='group':
        return _resnet('resnet18', GroupBasicBlock, [2, 2, 2, 2], pretrained, progress, prune_type,
                    **kwargs)
    elif prune_type =='filter':
        return _resnet('resnet18', FilterBasicBlock, [2, 2, 2, 2], pretrained, progress, prune_type,
                    **kwargs)




def resnet34(pretrained=False, progress=True, prune_type = None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        prune_type (str) : If None, not prune / choice filter or group
    """
    if prune_type is None:
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, prune_type,
                   **kwargs)
    elif prune_type =='group':
        return _resnet('resnet34', GroupBasicBlock, [3, 4, 6, 3], pretrained, progress, prune_type,
                    **kwargs)
    elif prune_type =='filter':
        return _resnet('resnet34', FilterBasicBlock, [3, 4, 6, 3], pretrained, progress, prune_type,
                    **kwargs)

    



def resnet50(pretrained=False, progress=True, prune_type = None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        prune_type (str) : If None, not prune / choice filter or group
    """
    if prune_type is None:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, prune_type,
                   **kwargs)
    elif prune_type =='group':
        return _resnet('resnet50', GroupBottleneck, [3, 4, 6, 3], pretrained, progress, prune_type,
                    **kwargs)
    elif prune_type =='filter':
        return _resnet('resnet50', FilterBottleneck, [3, 4, 6, 3], pretrained, progress, prune_type,
                    **kwargs)

    



def resnet101(pretrained=False, progress=True, prune_type = None, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        prune_type (str) : If None, not prune / choice filter or group
    """
    if prune_type is None:
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, prune_type,
                   **kwargs)
    elif prune_type =='group':
        return _resnet('resnet101', GroupBottleneck, [3, 4, 23, 3], pretrained, progress, prune_type,
                    **kwargs)
    elif prune_type =='filter':
        return _resnet('resnet101', FilterBottleneck, [3, 4, 23, 3], pretrained, progress, prune_type,
                    **kwargs)

    