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
    
    a2sf = torch.zeros_like(tensor[:,0,:])
    tmp_mask = torch.ones_like(tensor[:,0,:])
    
    for i in range(cache_budget):
        a2sf = penalty*a2sf + tensor[:,i,:]
    
    for i in range(cache_budget, tensor.shape[-2]):
        current_score = tensor[:,i,:]
        
        current_score *= tmp_mask
        current_score /= (torch.sum(current_score, dim=-1).unsqueeze(dim=-1) + 1e-10)
        
        if i != tensor.shape[-2]-1:
            if penalty != 0.0:
                a2sf = penalty*a2sf + current_score
            else:
                a2sf[a2sf!=torch.inf] = 0
                a2sf += current_score
        
            min_index = torch.argmin(a2sf[:,:i+1], axis=-1).unsqueeze(dim=-1)
            tmp_mask.scatter_(-1, min_index, 0)
            a2sf.scatter_(-1, min_index, np.inf)

    return tensor

def similarity(tensor_a, tensor_b):
    return torch.sum(torch.multiply(tensor_a, tensor_b))/(torch.norm(tensor_a)*torch.norm(tensor_b) + 1e-10)

model_name = "meta-llama/Llama-2-7b-hf"
penalty = 0.2
data_ratio = 0.01
search_iter = 100

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()

for dataset in ["openbookqa", "piqa", "arc_challenge", "arc_easy", "mathqa"]:
    file_path = f"/home/smp9898/A2SF/data/{dataset}-1shot.jsonl"

    with open(file_path, "r") as file:
        lines = file.readlines()

    data_size = int(len(lines) * data_ratio)
    result_ratio = None
    
    for _ in tqdm(range(data_size)):
        prompt = random.choice(lines)
        
        input_ids = tokenizer(get_prompt(prompt), add_special_tokens=True, return_tensors='pt').input_ids.cuda()

        with torch.no_grad():
            result = model(input_ids, output_attentions=True)

        tensors = torch.stack(result.attentions).squeeze(1)

        ratios = 0.2*torch.ones(tensors.shape[0])
        similarities = torch.tensor([similarity(tensors[i], make_mask(tensors[i], ratios[i], penalty)) for i in range(ratios.shape[0])])

        for i in range(search_iter):
            max_i = torch.argmax(similarities)
            min_i = torch.argmin(similarities)
            
            ratios[max_i] -= 0.01
            ratios[min_i] += 0.01
            
            similarities[max_i] = similarity(tensors[max_i], make_mask(tensors[max_i], ratios[max_i], penalty))
            similarities[min_i] = similarity(tensors[min_i], make_mask(tensors[min_i], ratios[min_i], penalty))

        if result_ratio is None:
            result_ratio = ratios
        else:
            result_ratio += ratios
        
        # for i in ratios: print(i.item(), end="\t")
        # print()
        
    result_ratio /= data_size

    print(dataset)
    for i in result_ratio: print(i.item())