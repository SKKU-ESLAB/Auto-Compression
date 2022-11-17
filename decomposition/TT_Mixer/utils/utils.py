import torch
import torch.nn as nn
from models.modules.tt_linear import TTLinear

def make_replaced_modules(model, target_idx, tt_configs):

    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) == nn.Linear:
            name_split = name.split('.')
            if ("channel_mlp_block" in name) and (int(name_split[1]) in target_idx):
                replaced_modules[name] = TTLinear(module,
                                                  tt_configs[module.in_features],
                                                  tt_configs[module.out_features],
                                                  tt_configs['ranks'])
    
    return replaced_modules
                
def replace_module_by_names(model, replaced_modules, tt_layers):

    target_name = []
    for name in tt_layers:
        name_split = name.split('.')
        target_name.append('.'.join(name_split[:-1]))

    def helper(childs: nn.Module):
        for name, child in childs.named_children():
            if type(child) is nn.Linear:
                for full_name, module in model.named_modules():
                    if (child is module) and (full_name in target_name):
                        childs.add_module(name, replaced_modules.pop(full_name))
                        break
            else:
                helper(child)
                
    helper(model)
    return model, replaced_modules

def make_target_name(model, target_idx):

    target_name = []
    for name, params in model.named_parameters():
        name_split = name.split('.')
        if ("channel_mlp_block" in name) and (int(name_split[1]) in target_idx):
            if "weight" in name:
                target_name.append(name)
                
    return target_name