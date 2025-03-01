import os
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from tqdm import tqdm

from utils.original_llama import LlamaForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The hash codes of q, need to be looked up in L hash tables, which is negligible computationally. However, the pre-built hash tables for kis can occupy considerable memory, making it a better fit for the CPU. With the above partition, we are able to support hash tables with K and L beyond the scale of prior work (Kitaev et al., 2020; Chen et al., 2021; Zandieh et al., 2023) without worrying about computation for hash codes as well as the storage of hash tables."

input_ids = tokenizer.encode(prompt, return_tensors="pt")

outputs = model(input_ids, output_attentions=True)

attentions = outputs.attentions

os.makedirs("analysis/weight", exist_ok=True)

layer_head_set = []
for i in range(32):
    for j in range(32):
        layer_head_set.append((i,j))
        
for i, j in tqdm(layer_head_set):
    tmp_attentions = attentions[i][0][0,j]
    tmp_attentions = torch.max(tmp_attentions, tmp_attentions.tril(0).min())
    
    plt.imshow(tmp_attentions.detach().cpu(), cmap="Blues")
    plt.xticks()
    plt.yticks()
    plt.savefig(f"analysis/weight/{i}_{j}.png")
    plt.close()