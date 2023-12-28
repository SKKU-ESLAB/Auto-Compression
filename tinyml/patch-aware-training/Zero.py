# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:42:44 2023

@author: kucha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
   
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

def find_modules_to_change(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 3:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.groups != 1:
                replaced_modules[n] = Mean_2px_function(m, 1, 2)
    

    return replaced_modules

def find_modules_to_change_first(model):
    replaced_modules = {}
    for n, m in model.named_modules():
        if len(replaced_modules) > 0:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3) and m.groups == 1:
                replaced_modules[n] = MyConv_function_first_stride2(m,2)
    

    return replaced_modules

class Mean_2px_function(nn.Module):
    def __init__(self, ConvModule, padding, num_patches):
        super(Mean_2px_function, self).__init__()
        self.conv = ConvModule
        self.conv.padding = (0, 0)
        self.padding = padding
        self.num_patches = num_patches

    def forward(self, x):
        x_patches = rearrange(x, "B C (ph H) (pw W) -> B ph pw C H W", ph=self.num_patches, pw=self.num_patches)
        x_patches = torch.nn.functional.pad(x_patches, (1, 1, 1, 1), mode='constant', value=0.)
        x_patches = rearrange(x_patches, "B ph pw C H W -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        out_patches = self.conv(x_patches)
        out_patches = rearrange(out_patches, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        # print(np_patches.shape)
        #print(out_patches)
        #plt.imshow(out_patches[0, 0, :, :].detach().numpy())
        #plt.show()
        #plt.axis('off')
        
        return out_patches
        
class MyConv_function_first_stride2(nn.Module):
    def __init__(self, ConvModule, num_patches):
        super(MyConv_function_first_stride2, self).__init__()
        self.mid_conv = ConvModule
        self.num_patches = num_patches
        self.mid_conv.padding = (0, 0)
        self.groups = ConvModule.groups
        #self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        #nn.init.kaiming_normal_(self.border_weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        B, C, H, W = x.size()
        padded_x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        overlap_patches = padded_x.unfold(2,  H//self.num_patches + 2, H//self.num_patches).unfold(3, W//self.num_patches + 2, W//self.num_patches)
        overlap_patches = rearrange(overlap_patches, 'B C  ph pw H W -> (B ph pw) C H W', ph=self.num_patches, pw=self.num_patches)
        out = self.mid_conv(overlap_patches)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out   
             
Module_mapping = {
    nn.Conv2d : Mean_2px_function
}

Module_first_mapping = {
    nn.Conv2d : MyConv_function_first_stride2
}

        
if __name__ == "__main__":
    conv = nn.Conv2d(3, 3, padding=0, kernel_size=3, bias = False)
    nn.init.constant(conv.weight, 1.)
    func = Mean_2px_function(conv, padding = 1, num_patches = 4)

    # inx = torch.randn((1, 3, 224, 224))
    inx = torch.ones(1, 3, 112, 112)
    print(func)
    func(inx)

#%%
import torch

a = torch.arange(5)
print(a)
print(a[:-1])
