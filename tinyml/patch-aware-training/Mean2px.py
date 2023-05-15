# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:42:44 2023

@author: kucha
"""

import torch
import torch.nn as nn
from einops import rearrange
   
def replace_module_by_names(model, modules_to_replace):
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
        if len(replaced_modules) > 4:
            break
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3):
                replaced_modules[n] = Mean_2px_function(m, 1, 4)
    

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
        pad_top = x_patches[:, :, :, :, :2, :].mean(dim=-2, keepdim=True)
        #pad_top[:, :, :, :, :, :] *= 0

        pad_top[:, 0, :, :, :, :] *= 0.
        
        pad_down = x_patches[:, :, :, :, -2:, :].mean(dim=-2, keepdim=True)
        
        #pad_down[:, :, :, :, :, :] *= 0
        
        pad_down[:, -1, :, :, :, :] *= 0.

        pad_left = x_patches[:, :, :, :, :, :2].mean(dim=-1, keepdim=True)
        
        #pad_left[:, :, :, :, : :] *= 0.

        pad_left[:, :, 0, :, :, :] *= 0.

        pad_right = x_patches[:, :, :, :, :, -2:].mean(dim=-1, keepdim=True)
        
        #pad_right[:, :, :, :, :, :] *=0.
        pad_right[:, :, -1, :, :, :] *= 0.
        
        pad_topleft = x_patches[:, :, :, :, 0, 0].clone().unsqueeze(dim = (-1)).unsqueeze(-1)
        
        #pad_topleft[:, :, :, :, :, :] *= 0.

        pad_topleft[:, 0, :, :, :, :] *= 0.
        pad_topleft[:, :, 0, :, :, :] *= 0.

       
        pad_topright = x_patches[:, :, :, :, 0, -1].clone().unsqueeze(dim = (-1)).unsqueeze(-1)
        #pad_topright[:, :, :, :, :, :] *= 0.

        pad_topright[:, 0, :, :, :, :] *= 0.
        pad_topright[:, :, -1, :, :, :] *= 0.

        pad_botleft = x_patches[:, :, :, :, -1, 0].clone().unsqueeze(dim = (-1)).unsqueeze(-1)
        
        #pad_botleft[:, :, :, :, :, :] *= 0

        pad_botleft[:, -1, :, :, :, :] *= 0
        pad_botleft[:, :, 0, :, :, :] *= 0

        pad_botright = x_patches[:, :, :, :, -1, -1].clone().unsqueeze(dim = (-1)).unsqueeze(-1)
        
        #pad_botright[:, :, :, :, :, :] *= 0.

        pad_botright[:, :, -1, :, :, :] *= 0
        pad_botright[:, -1, :, :, :, :] *= 0

        #pad_top *= 0.
        #pad_down *= 0.
        #pad_left *= 0.
        #pad_right *= 0.
        #pad_topleft *=0.
        #pad_topright *=0.
        #pad_botleft *= 0.
        #pad_botright *=0.
        
        pad_ex_top = torch.cat([pad_topleft, pad_top, pad_topright], dim=-1)
        pad_ex_down = torch.cat([pad_botleft, pad_down, pad_botright], dim=-1)
        x_patches = torch.cat([pad_left, x_patches, pad_right], dim = -1)

        x_patches = torch.cat([pad_ex_top, x_patches, pad_ex_down], dim = -2)
        #x_patches = torch.nn.functional.pad(x_patches, (1, 1, 1, 1), mode='constant', value=0.0) 
        x_patches = rearrange(x_patches, "B ph pw C H W -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        out_patches = self.conv(x_patches)
        out_patches = rearrange(out_patches, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        # print(np_patches.shape)
        #print(out_patches)
        #plt.imshow(out_patches[0, 0, :, :].detach().numpy())
        #plt.show()
        #plt.axis('off')
        
        return out_patches
        
Module_mapping = {
    nn.Conv2d : Mean_2px_function
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
