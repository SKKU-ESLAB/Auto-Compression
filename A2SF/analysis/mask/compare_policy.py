import copy
import torch
import os
import random
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

sys.path.append("/home/smp9898/A2SF")

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
from utils_lm_eval.ideal_llama import convert_kvcache_llama_heavy_recent_ideal

def get_prompt(json_line):
    data = json.loads(json_line)
    return data["prompt"]

def compare_mask(mask1, mask2):
    result = list()
    
    for i in range(mask1.shape[1]):
        mask_a = mask1[0, i, :, :]
        mask_b = mask2[0, i, :, :]
        similar = torch.sum(torch.multiply(mask_a, mask_b)/(torch.norm(mask_a) * torch.norm(mask_b) + 1e-10))

        result.append(similar.item())
        
    return sum(result)/len(result)

model_name = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()
check_point = copy.deepcopy(model.state_dict())
model.cuda()

root_path = os.path.dirname(__file__)

datasets = ["piqa", "openbookqa", "arc_easy", "arc_challenge", "mathqa"]
prompts = []

seq_length = 0
for dataset in datasets:
    file_path = f"/home/smp9898/A2SF/data/{dataset}-1shot.jsonl"
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for i in range(4):
        prompt = get_prompt(random.choice(lines))
        input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids
        seq_length += input_ids.numel()
        prompts.append(input_ids.cuda())

averaged_seq_length = seq_length/len(prompts)
print(f"average seq length {averaged_seq_length:.2f}")

full_masks = []
with tqdm(prompts) as pbar:
    pbar.set_description("Full model mask making")
    for prompt in pbar:
        with torch.no_grad():
            result = model(prompt, output_attentions=True)

        full_masks.append([result.attentions[layer].cpu().detach().to(torch.float) for layer in range(len(result.attentions))])

num_layers = 32
ratio = 0.1

methods = {
    "LOCAL": (0.0, 0.0, ratio, 1.0, False),
    # "STREAMING_LLM": (ratio/2, 0.0, ratio/2, 1.0, False),
    "H2O": (0.0, ratio/2, ratio/2, 1.0, False),
    "A2SF": (0.0, ratio, 0.0, 0.2, False),
}

mask_list = {
    "LOCAL": np.zeros(num_layers),
    # "STREAMING_LLM": np.zeros(num_layers),
    "H2O": np.zeros(num_layers),
    "A2SF": np.zeros(num_layers),
}

for name, (i, j, k, h, ideal) in methods.items():
    config.streaming_ratio = i
    config.selecting_ratio = j
    config.recent_ratio = k
    config.forgetting_factor = h
    config.tmp = None
    
    if ideal:
        convert_kvcache_llama_heavy_recent_ideal(model, config)
    else:
        convert_kvcache_llama_heavy_recent(model, config)
    
    model.load_state_dict(check_point)
    torch.cuda.empty_cache()
    model.half().eval().cuda()
    
    tmp_masks = []
    with tqdm(enumerate(prompts)) as pbar:
        pbar.set_description(name)
        for index, prompt in pbar:
            with torch.no_grad():
                result = model(prompt, output_attentions=True)
            
            tmp_masks.append([result.attentions[layer].cpu().detach().to(torch.float) for layer in range(len(result.attentions))])
    
    for index in tqdm(range(len(prompts))):
        for layer in range(num_layers):
            mask_list[name][layer] += compare_mask(full_masks[index][layer], tmp_masks[index][layer])

folder_path = os.path.join(root_path, "similarity")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

plt.figure(figsize=(7.5,6))
for a, b in mask_list.items():
    averaged_b = b/len(prompts)
    if np.sum(b) != 0:
        plt.plot(averaged_b, label=a)
        
        mean = np.mean(averaged_b)
        print(f"{a} : {mean:.3f}")

plt.legend()
plt.title(f"Average Cosine Similarity of Heads")
plt.xlabel("Layer Number")
plt.ylabel("Similarity")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f"similarity.png"))
plt.close()