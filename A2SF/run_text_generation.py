import argparse
import logging

import numpy as np
import torch
import json
import copy
import random

from rouge import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_real_drop.modify_llama import H2OLlamaAttention
tm
def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

def make_text(model, tokenizer, input_ids, answer):
    generate_ids = model.generate(input_ids, max_new_tokens=64, do_sample=False, temperature=1.0, top_p=1.0)
    result = tokenizer.batch_decode(generate_ids)[0]
    result = result.replace(prompt_text, "")
    result = result[:result.find("###")].strip()
    print(result)
    print()
    score = rouge.get_scores(result, answer, avg=True)
    print(score)
    print()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--cache_budget", type=int, default=50)
    parser.add_argument("--forgetting_factor", type=float, default=0.4)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {args.device}")

    rouge = Rouge()

    with open("data/xsum-3shot.jsonl", "r") as f:
        file = f.readlines()
        data = json.loads(random.choice(file))
        prompt_text = data["article"]
        answer = data["summary_gt"]

    model_name = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()
    
    check_point = copy.deepcopy(model.state_dict())
    model.to(args.device)

    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
    print(f"prompt length: {input_ids.shape}")
    print(prompt_text)
    print(f"*** {answer} ***")
    print()

    ######## Generate with Full Cache
    print("######################### Full Cache Model #########################")
    make_text(model, tokenizer, input_ids, answer)

    ######### Enable H2O
    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))
    config.scoring_policy = "h2o"
    config.hh_size = int(args.cache_budget/2)
    config.recent_size = int(args.cache_budget/2)
    config.forgetting_factor = 1.0
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn = H2OLlamaAttention(config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    print("######################### H2O Model #########################")
    make_text(model, tokenizer, input_ids, answer)

    ######### Enable A2SF
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn.kv_cache.scoring_policy = "a2sf"
        model.model.layers[layer_idx].self_attn.kv_cache.forgetting_factor = args.forgetting_factor
        model.model.layers[layer_idx].self_attn._clean_cache()

    print("######################### A2SF Model #########################")
    make_text(model, tokenizer, input_ids, answer)
