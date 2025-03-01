import copy
import torch
import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

dir_path = os.path.dirname(__file__)

def mode(x: list):
    tmp_dict = {}
    for i in set(x):
        tmp_dict[i] = 0
    for i in x:
        tmp_dict[i] += 1
    return tmp_dict

def get_prompt(json_line):
    data = json.loads(json_line)
    # return data["prompt"]
    return data["article"] 

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

# def similarity(tensor_a, tensor_b):
#     norms = torch.norm(tensor_a, dim=(-1,-2)) * torch.norm(tensor_b, dim=(-1,-2)) + 1e-10
#     headwise_similarity = torch.sum(torch.multiply(tensor_a, tensor_b), dim=(-1,-2))/norms
#     return headwise_similarity

def similarity(vectors_a, vectors_b):
    norms = torch.norm(vectors_a, dim=-1) * torch.norm(vectors_b, dim=-1) + 1e-10
    tokenwise_similarity = torch.sum(torch.multiply(vectors_a, vectors_b), dim=-1)/norms
    return tokenwise_similarity

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "huggyllama/llama-7b"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

config.streaming_ratio = 0.0
config.selecting_ratio = 0.1
config.recent_ratio = 0.0
config.forgetting_factor = 0.3
config.tmp = None

ratios = 0.2
data_ratio = 0.1
penalties = torch.arange(0.0, 1.0, 0.1)
simlir = None

plt.figure(figsize=(9,6))

model = model.half().eval().cuda()
for idx, dataset in enumerate(["xsum"]):
    file_path = f"/home/smp9898/A2SF/data/{dataset}-3shot.jsonl"

    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # similarities = torch.zeros_like(penalties)
    # data_size = int(data_ratio * len(lines))
    data_size = 100
    asdf = []
    
    with tqdm(range(data_size)) as pbar:
        pbar.set_description(dataset)
        for _ in pbar:
            prompt = random.choice(lines)
            
            input_ids = tokenizer(get_prompt(prompt), add_special_tokens=True, return_tensors='pt').input_ids.cuda()
            
            pbar.set_postfix({"length": input_ids.numel()})
            
            with torch.no_grad():
                result = model(input_ids, output_attentions=False)
    #         tensors = torch.stack(result.attentions).squeeze(1).detach().to(torch.float)
    #         values = torch.stack([result.past_key_values[i][1] for i in range(32)]).squeeze(1).detach().to(torch.float)

    #         full_values = torch.matmul(tensors, values).transpose(1,2).reshape(*tensors.shape[:2], -1)

    #         tmp = []
    #         for penalty in penalties:
    #             masked_tensors = make_mask(tensors, ratios, penalty)
    #             masked_values = torch.matmul(masked_tensors, values).transpose(1,2).reshape(*full_values.shape)
                
    #             tmp.append(torch.mean(similarity(full_values, masked_values)).item())

    #         asdf.append(tmp)
    
    # asdf = torch.tensor(asdf)
    
    # if simlir is None:
    #     simlir = asdf.sum(0)
    # else:
    #     simlir += asdf.sum(0)
    
    # # torch.save(asdf, os.path.join(dir_path, f"{dataset}.pt"))
    
    # print(dataset)
    # # print(mode(torch.max(asdf, dim=-1).indices.tolist()))
    # print(asdf.sum(0).argmax())
    # print(simlir.argmax())

# for i in range(similarities.shape[0]): print(f"{penalties[i]:.2f} {similarities[i]:.4f}")
# print(f"{penalties[torch.argmax(similarities)]:.2f}")

# # plt.subplot(2, 3, idx+1)
# # plt.title(f"{dataset} : {penalties[torch.argmax(similarities)]:.2f}")
# plt.plot(penalties, similarities)

# plt.tight_layout()
# plt.savefig(os.path.join(dir_path, f"forgetting_factor_fixed_ratio_020_5_shots.png"))