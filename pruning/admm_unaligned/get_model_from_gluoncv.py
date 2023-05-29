#import mxnet as mx
import torch
import gluoncv
import torch.nn as nn
import torchvision.models as models
import numpy as np

from models.mobilenet_v1 import MobileNetV1
#from models.mobilenet_v2 import MobileNetV2

def get_pretrained_model(arch, width_mult):
    # you may modify it to switch to another model. The name is case-insensitive
    if arch == "mobilenet_v1":
        model_name = "mobilenet" + str(width_mult)
        torch_model = MobileNetV1(num_classes=1000, input_size=224, width_mult=width_mult)
    elif arch == "mobilenet_v2":
        model_name = "mobilenetv2_" + str(width_mult)
        #torch_model = MobileNetV2(num_classes=1000, input_size=224, width_mult=width_mult)
        torch_model = models.mobilenet_v2(width_mult=width_mult)
    # download and load the pre-trained model
    mxnet_model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    #print(mxnet_model)
