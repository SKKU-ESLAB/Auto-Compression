import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torchvision.models._utils import _make_divisible
from torchvision.ops import SqueezeExcitation
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
    def __init__(self, ConvModule, padding,  num_patches):
        super(MyConv_function_stride1, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.padding = padding
        #self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        # self.kw = self.mid_conv.kernel_size[0]
        self.border_conv_region = self.mid_conv.kernel_size[0] + self.padding - 1
        print(self, self.num_patches)

    def extra_repr(self):
        s = (f'(Conv warpper): no-patch-seperate , num-patches={self.num_patches}, padding={self.padding}')
        return s.format(**self.__dict__)
    
    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error'
        padded_x_patches = F.pad(x_patches, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0.0)
        out = self.mid_conv(padded_x_patches)
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        
        return out

class MyConv_function_stride2_kernel3(nn.Module):
    def __init__(self, ConvModule, padding,  num_patches):
        super(MyConv_function_stride2_kernel3, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.padding = padding
        #self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        print(self, self.num_patches)

    def forward_odd(self, patches):
        padded_patches = F.pad(patches, (1, 1, 1, 1), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out
    
    def forward_even(self, patches):
        padded_patches = F.pad(patches, (1, 1, 1, 1), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out
    
    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error' 
        is_odd = SH % 2 == 1
        
        if is_odd:
            out = self.forward_odd(x_patches)
        else:
            out = self.forward_even(x_patches)
            
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out     
        
    def extra_repr(self):
        s = (f'(Conv warpper): no-patch-seperate , num-patches={self.num_patches}, padding={self.padding}')
        return s.format(**self.__dict__)
        
    
class MyConv_function_stride2_kernel5(nn.Module):
    def __init__(self, ConvModule, padding,  num_patches):
        super(MyConv_function_stride2_kernel5, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.padding = padding
        #self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        print(self, self.num_patches)

    def forward_odd(self, patches):
        padded_patches = F.pad(patches, (2, 2, 2, 2), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out
    
    def forward_even(self, patches):
        padded_patches = F.pad(patches, (2, 2, 2, 2), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out 
    
    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error' 
        is_odd = SH % 2 == 1
        
        if is_odd:
            out = self.forward_odd(x_patches)
        else:
            out = self.forward_even(x_patches)
            
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out     

    def extra_repr(self):
        s = (f'(Conv warpper): no-patch-seperate , num-patches={self.num_patches}, padding={self.padding}')
        return s.format(**self.__dict__)
    
class MySqueezeExcitation(nn.Module):
    def __init__(self, SEBlock, num_patches):
        super(MySqueezeExcitation, self).__init__()
        self.avgpool = SEBlock.avgpool
        self.fc1 = SEBlock.fc1
        self.fc2 = SEBlock.fc2
        self.activation = SEBlock.activation
        self.scale_activation = SEBlock.scale_activation
        self.num_patches = num_patches
        print(self, self.num_patches)

    def _scale(self, input):
        # print(input.size())
        scale = self.avgpool(input)
        # print(scale.size())
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
    
    def forward(self, x):
        # print(x.size())
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)        
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error'
        patch_scale = self._scale(x_patches)
        out = patch_scale * x_patches
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out
        
    def extra_repr(self):
        s = (f'(SE Block warpper): num-patches={self.num_patches}')
        return s.format(**self.__dict__)

class MyConv_function_stride2_kernel7(nn.Module):
    def __init__(self, ConvModule, padding,  num_patches):
        super(MyConv_function_stride2_kernel7, self).__init__()
        self.mid_conv = ConvModule
        self.mid_conv.padding = (0, 0)
        self.num_patches = num_patches
        self.groups = ConvModule.groups
        self.padding = padding
        #self.border_weight = torch.nn.Parameter(self.mid_conv.weight.detach())
        print(self, self.num_patches)


    def forward_odd(self, patches):
        padded_patches = F.pad(patches, (3, 3, 3, 3), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out
    
    def forward_even(self, patches):
        padded_patches = F.pad(patches, (3, 3, 3, 3), mode='constant', value=0.0)
        out = self.mid_conv(padded_patches)
        
        return out

    def forward(self, x):
        _, _, LH, LW = x.size()
        x_patches = rearrange(x, "B C (ph H) (pw W) -> (B ph pw) C H W", ph=self.num_patches, pw=self.num_patches)
        _, _, SH, SW = x_patches.size()
        assert LH == SH * self.num_patches, 'error' 
        is_odd = SH % 2 == 1
        
        if is_odd:
            out = self.forward_odd(x_patches)
        else:
            out = self.forward_even(x_patches)
            
        out = rearrange(out, "(B ph pw) C H W -> B C (ph H) (pw W)", ph=self.num_patches, pw=self.num_patches)
        return out    
    
    def extra_repr(self):
        s = (f'(Conv warpper): no-patch-seperate , num-patches={self.num_patches}, padding={self.padding}')
        return s.format(**self.__dict__)
module_to_mapping = {
    (3, 3, 1): MyConv_function_stride1,
    (5, 5, 1): MyConv_function_stride1,
    (7, 7, 1): MyConv_function_stride1,
    (3, 3, 2): MyConv_function_stride2_kernel3,
    (5, 5, 2): MyConv_function_stride2_kernel5,
    (7, 7, 2): MyConv_function_stride2_kernel7
}

def is_spatial(layer):
    if isinstance(layer, nn.Conv2d):
        k = None
        if isinstance(layer.kernel_size, tuple):
            k = layer.kernel_size[0]
        else:
            pass
        if k > 1:
            return True
        return False
    return False

def get_stride_layer(layer):
    s = layer.stride
    if isinstance(s, int):
        s = (s, s)
    return s

def get_attr(layer):
    k = layer.kernel_size
    s = layer.stride

    if isinstance(k, int):
        k = (k, k)
    if isinstance(s, tuple):
        s = s[0]
    return (*k, s)

def change_model(model, num_patches, Module_To_Mapping, num_per_patch_stage):
    i = 0
    for name, target in model.named_modules():
        if i == num_per_patch_stage:
            return model
        if is_spatial(target):
            i += 1
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            pad = target.padding
            if isinstance(target.padding, tuple):
                pad = pad[0]
            replace = Module_To_Mapping[get_attr(target)](target, pad, num_patches)
            setattr(submodule, attrs[-1], replace)
            
def replace_layer(model, block_id, patch_type, modules_to_replace):
    if patch_type == 1:
        return
    layers = model.features[block_id]
    target = layers

    for name, layer in layers.named_modules():
        if is_spatial(layer):
            attrs = name.split('.')
            for attr in attrs[:-1]:
                target = getattr(target, attr)

            check_stride = get_stride_layer(layer)
            replace = modules_to_replace[check_stride](layer, layer.padding[0], 4)
            setattr(target, attrs[-1], replace)

def change_model_with_se(model, num_patches, Module_To_Mapping, num_per_patch_stage):
    i = 0
    for name, target in model.named_modules():
        # print(model)
        if i == num_per_patch_stage:
            break
        if is_spatial(target):
            attrs = name.split('.')
            submodule = model
            for attr in attrs[:-2]:
                submodule = getattr(submodule, attr)
            SEBlock = None
            SEBlock_id = None
            # print(submodule)
            # print(name)
            for name_, module in submodule.named_modules():
                if isinstance(module, SqueezeExcitation):
                    SEBlock_id = name_
                    SEBlock = module
                    break
                    # print(name_)
            if SEBlock_id != None:
                # print(submodule)
                replace_SEBlock = MySqueezeExcitation(SEBlock, num_patches)
                submodule_se = submodule
                se_block_attr =  SEBlock_id.split('.')
                for se_attr in SEBlock_id.split('.')[:-1]:
                    submodule_se = getattr(submodule_se, se_attr)
                setattr(submodule_se, se_block_attr[-1], replace_SEBlock)
                # print('\n\n 변경이후')
                # print(submodule)
                # print(submodule)
            # print(model)
            submodule = getattr(submodule, attrs[-2])
            pad = target.padding
            if isinstance(target.padding, tuple):
                pad = pad[0]
            replace =  Module_To_Mapping[get_attr(target)](target, pad, num_patches)
            # print(submodule, '\n', target)
            setattr(submodule, attrs[-1], replace)   
            # print(submodule)
            i += 1
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
