import torch
import torchvision
import torch.backends.cudnn as cudnn
import os
from models import *

def model_builder(model, dataset, is_qt=False, lq=False, index=None, fwlq=False):

    if dataset == 'cifar10':
        _class = 10
    elif dataset == 'cifar100':
        _class = 100
    else:
        _class = 1000

    if model == 'alexnet':
        net = alexnet_cifar(num_classes=_class, is_qt=is_qt, lq=lq) \
                if 'cifar' in dataset else alexnet(lq=lq)
    elif model == 'mobilenetv2':
        net = mobilenetv2_cifar(num_classes=_class, is_qt=is_qt, lq=lq, index=index, fwlq=fwlq) \
                if 'cifar' in dataset else mobilenetv2(lq=lq, index=index)
    
    elif model == 'resnet20':
        net = resnet20(is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=_class)
    elif model == 'resnet32':
        net = resnet32(is_qt=is_qt, lq=lq, fwlq=fwlq, num_classes=_class)
    
    elif model == 'resnet18':
        net = resnet18(lq=lq, num_classes=_class)
    elif model == 'resnet34':
        net = resnet32(lq=lq, num_classes=_class)

    else:
        assert False, 'No Such Model'



    return net
