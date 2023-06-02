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

    param_list = [param._reduce().asnumpy() for name, param in mxnet_model.collect_params().items()]

    for m in torch_model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.from_numpy(param_list.pop(0))
            if m.bias is not None:
                m.bias.data = torch.from_numpy(param_list.pop(0))
        elif isinstance(m, nn.Linear):
            if arch == "mobilenet_v2":
                m.weight.data = torch.from_numpy(param_list.pop(0).squeeze())
                m.bias.data.fill_(0)
            else:
                m.weight.data = torch.from_numpy(param_list.pop(0))
                if m.bias is not None:
                    m.bias.data = torch.from_numpy(param_list.pop(0))
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                m.weight.data = torch.from_numpy(param_list.pop(0))
                m.bias.data = torch.from_numpy(param_list.pop(0))
            if m.track_running_stats:
                m.running_mean.data = torch.from_numpy(param_list.pop(0))
                m.running_var.data = torch.from_numpy(param_list.pop(0))


    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    torch_model = torch.nn.DataParallel(torch_model).cuda()
    save_checkpoint({
        'admm_epoch': 0,
        'ft_epoch': 0,
        'arch': arch,
        'state_dict': torch_model.state_dict(),
        'best_acc1': 0.,
        'optimizer': None,
        'perm_list': None,
        'mask': None,
    }, False, filename=f'{arch}_{width_mult}.pth.tar')


if __name__ == "__main__":
    get_pretrained_model("mobilenet_v2", 1.0)
