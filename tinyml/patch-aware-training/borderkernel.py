import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torchvision.models._utils import _make_divisible
import numpy as np
import os

def replace_module_by_names(model, modules_to_replace, Module_mapping):
    def helper(child: nn.Module):
        for n, c in child.named_children():
            if type(c) in Module_mapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)
    helper(model)
    return model

def find_modules_to_change_stride2(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 2:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.stride == (2, 2):
                replaced_modules[n] = MyConv_function_stride2(m, 4)
    

    return replaced_modules

def find_modules_to_change_stride1(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 1:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.stride == (1, 1):
                replaced_modules[n] = MyConv_function_stride1(m, 4)
    

    return replaced_modules




class MyConv_function_stride1(nn.Module):
    def __init__(self, ConvModule, num_patches):
        super(MyConv_function_stride1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        
    def forward(self, x):
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        padded_x_patches = F.pad(x_patches, (1, 1, 1, 1), mode='constant', value=0.0)
        Mid = self.mid_conv(x_patches)
        out_top = F.conv2d(padded_x_patches[:, :, :3, 1:-1], self.border_weight, groups=self.groups)
        out_bot = F.conv2d(padded_x_patches[:, :, -3:, 1:-1], self.border_weight, groups=self.groups)
        out_left = F.conv2d(padded_x_patches[:, :, 1:-1, :3], self.border_weight, groups=self.groups)
        out_right = F.conv2d(padded_x_patches[:, :, 1:-1, -3:], self.border_weight, groups=self.groups)
    
        out_top_left =F.conv2d(padded_x_patches[:, :, :3, :3], self.border_weight, groups=self.groups)
        out_top_right =F.conv2d(padded_x_patches[:, :, :3, -3:], self.border_weight, groups=self.groups)
        out_bot_left =F.conv2d(padded_x_patches[:, :, -3:, :3], self.border_weight, groups=self.groups)
        out_bot_right =F.conv2d(padded_x_patches[:, :, -3:, -3:], self.border_weight, groups=self.groups)
        
        out_ex_top = torch.cat([out_top_left, out_top, out_top_right], dim=-1)
        out_ex_down = torch.cat([out_bot_left, out_bot, out_bot_right], dim=-1)
        Mid = torch.cat([out_left, Mid, out_right], dim = -1)
        out = torch.cat([out_ex_top, Mid, out_ex_down], dim = -2)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out
    
class MyConv_function_stride2(nn.Module):
    def __init__(self, ConvModule, num_patches):
        super(MyConv_function_stride2, self).__init__()
        self.mid_conv = ConvModule
        self.num_patches = num_patches
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        
    def forward(self, x):
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        padded_x_patches = F.pad(x_patches, (1, 0, 1, 0), mode='constant', value=0.0)
        Mid = self.mid_conv(x_patches[:, :, 1:, 1:])
        
        out_top = F.conv2d(padded_x_patches[:, :, :3, 2:], self.border_weight, groups=self.groups, stride=2)
        # print(Mid.size(), out_top.size())

        out_left = F.conv2d(padded_x_patches[:, :, 2:, :3], self.border_weight, groups=self.groups, stride=2)
    
        out_top_left =F.conv2d(padded_x_patches[:, :, :3, :3], self.border_weight, groups=self.groups, stride=2)
        

        Mid = torch.cat([out_left, Mid], dim=-1)
        out_ex_top = torch.cat([out_top_left, out_top], dim=-1)
        out = torch.cat([out_ex_top, Mid], dim=-2)
        
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out
    
Module_stride1_mapping = {
    nn.Conv2d : MyConv_function_stride1
}

Module_stride2_mapping = {
    nn.Conv2d : MyConv_function_stride2
}



#model = mobilenet_v2(pretrained=True)

#modules_to_replace = find_modules_to_change_stride1(model)
# print(modules_to_replace)
#replace_module_by_names(model, modules_to_replace, Module_stride1_mapping)
#modules_to_replace = find_modules_to_change_stride2(model)
# print(modules_to_replace)
#replace_module_by_names(model, modules_to_replace, Module_stride2_mapping)
#print(model)
#model(torch.randn(1, 3, 224, 224))

#%%
#a = [1, 2, 3, 4, 5]
#print(a[2:])
