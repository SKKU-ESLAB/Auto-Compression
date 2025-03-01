import copy
import torch

from typing import Union

from transformers import AutoModelForCausalLM, AutoConfig

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
from lm_eval.models.huggingface import HFLM

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "OPTForCausalLM": convert_kvcache_opt_heavy_recent,
    "LlamaForCausalLM": convert_kvcache_llama_heavy_recent,
}

def lm_model(model_name: str,
             lm: HFLM=None,
             check_point=None,
             device="cpu",
             streaming_ratio: float = 0.0,
             selecting_ratio: float = 0.1,
             recent_ratio: float = 0.1,
             layerwise_ratio: list = None,
             forgetting_factor: float = 1.0,
             tmp: int = None,
             ideal: bool=False,
            ):
    config = AutoConfig.from_pretrained(model_name)
    config.streaming_ratio = streaming_ratio
    config.selecting_ratio = selecting_ratio
    config.recent_ratio = recent_ratio
    config.forgetting_factor = forgetting_factor
    config.layerwise_ratio = layerwise_ratio
    config.tmp = tmp
    config.masking_mode = "fas"

    lm.model.cpu()

    if ideal:
        arch = "IdealLlamaForCausalLM"
    else:
        arch = config.architectures[0]

    ENABLE_Heavy_Hitter_FUNCTIONS[arch](lm.model, config=config)

    lm.model.load_state_dict(check_point)
    
    lm.model.eval().half().to(device)