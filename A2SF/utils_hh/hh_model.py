import copy
import torch

from typing import Union

from transformers import AutoModelForCausalLM, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent
from lm_eval.models.huggingface import HFLM

def hh_model(model_name: str,
             lm: HFLM=None,
             heavy_ratio: float = 0.1,
             recent_ratio: float = 0.1,
             penalty: float = 1.0,
             cache_dir = None
            ):
    config = AutoConfig.from_pretrained(model_name)
    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio
    config.penalty = penalty

    if lm is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        model = lm.model.cpu()

    checkpoint = copy.deepcopy(model.state_dict())

    if (config.architectures[0] == "OPTForCausalLM"):
        model = convert_kvcache_opt_heavy_recent(model, config=config)
    elif (config.architectures[0] == "LlamaForCausalLM"):
        model = convert_kvcache_llama_heavy_recent(model, config=config)

    model.load_state_dict(checkpoint)

    if lm is None:
        return model

def reset_mask(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = reset_mask(module)

        if hasattr(module, "_reset_masks"):
            module._reset_masks()
    return model