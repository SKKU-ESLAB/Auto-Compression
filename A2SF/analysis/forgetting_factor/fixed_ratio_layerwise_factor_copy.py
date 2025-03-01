import copy
import torch
import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def get_prompt(json_line):
    data = json.loads(json_line)
    return data["prompt"]

def make_mask(input_tensor, heavy_ratio, penalty):
    tensor = torch.clone(input_tensor)
    cache_budget = int(tensor.shape[-2]*heavy_ratio)
    
    a2sf = torch.zeros_like(tensor[:,:,0,:])
    tmp_mask = torch.ones_like(tensor[:,:,0,:])
    
    for i in range(cache_budget):
        a2sf = penalty*a2sf + tensor[:,:,i,:]
    
    for i in range(cache_budget, tensor.shape[-2]):
        current_score = tensor[:,:,i,:]
        
        current_score *= tmp_mask
        current_score /= (torch.sum(current_score, dim=-1).unsqueeze(dim=-1) + 1e-10)
        
        if i != tensor.shape[-2]-1:
            if penalty != 0.0:
                a2sf = penalty*a2sf + current_score
            else:
                a2sf[a2sf!=torch.inf] = 0
                a2sf += current_score
        
            min_index = torch.argmin(a2sf[:,:,:i+1], axis=-1).unsqueeze(dim=-1)
            tmp_mask.scatter_(-1, min_index, 0)
            a2sf.scatter_(-1, min_index, np.inf)

    return tensor

def similarity(tensor_a, tensor_b):
    return torch.sum(torch.multiply(tensor_a, tensor_b))/(torch.norm(tensor_a)*torch.norm(tensor_b) + 1e-10)

model_name = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()

for idx, dataset in enumerate(["openbookqa", "piqa", "arc_challenge", "arc_easy", "mathqa"]):
    file_path = f"/home/smp9898/A2SF/data/{dataset}-1shot.jsonl"

    with open(file_path, "r") as file:
        lines = file.readlines()

    ratios = 0.2
    penalties = torch.arange(0.0, 1.0, 0.05)
    similarities = torch.zeros((penalties.shape[0], config.num_hidden_layers))
    data_ratio = 0.01
    data_size = int(len(lines) * data_ratio)

    with tqdm(range(data_size)) as pbar:
        pbar.set_description(dataset)
        for _ in pbar:
            prompt = random.choice(lines)
            
            input_ids = tokenizer(get_prompt(prompt), add_special_tokens=True, return_tensors='pt').input_ids.to(model.device)

            with torch.no_grad():
                result = model(input_ids, output_attentions=True)

            tensors = torch.stack(result.attentions).squeeze(1)

            for row, penalty in enumerate(penalties):
                masked_tensors = make_mask(tensors, ratios, penalty)
                tmp = torch.tensor([similarity(tensors[i], masked_tensors[i]) for i in range(config.num_hidden_layers)])
                if torch.any(torch.isnan(tmp)):
                    continue   
                similarities[row] += tmp
        
            max_panalties = torch.argmax(similarities, dim=0)
            for i in range(config.num_hidden_layers): print(f"{penalties[max_panalties[i]]:.2f}", end="\t")
            print()
            similarities = torch.zeros((penalties.shape[0], config.num_hidden_layers))

    # max_panalties = torch.argmax(similarities, dim=0)

    # print(dataset)
    # for i in range(config.num_hidden_layers): print(f"layer_{i} {penalties[max_panalties[i]]:.2f}")