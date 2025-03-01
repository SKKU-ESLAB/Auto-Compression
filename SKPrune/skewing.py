import torch
import os
import matplotlib.pyplot as plt
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.colors import LogNorm
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Model Pruning with Skewed Key Vectors")
parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for pruning")
args = parser.parse_args()

threshold = args.threshold

device = "cuda:1"
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

input_ids = torch.randint(0, 50272, (1, 2048)).to(model.device)

with torch.no_grad():
    outputs = model(input_ids)
    key_values = outputs.past_key_values

    dummy_input = "A tranquil mountain landscape at dawn, with mist hovering over a crystal-clear lake surrounded by tall pine trees, while the first light of the day gently illuminates the snow-capped peaks in the distance."
    input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids.to(model.device)
    outputs = model(input_ids)

    layerwise_pruning_ratio = []

    for layer_idx, (key, value) in enumerate(tqdm(key_values)):
        key = key.squeeze(0)
        V = key.svd(some=False).V
        
        skewed_key = torch.matmul(key, V)
        skewed_key = skewed_key.view(skewed_key.size(1),-1)
        std = skewed_key.std(dim=0)
        mask = (std > threshold).float()
        
        for head_idx in range(key.size(0)):
            model.model.decoder.layers[layer_idx].self_attn.q_proj.weight[80*(head_idx):80*(head_idx+1)].copy_(mask[80*(head_idx):80*(head_idx+1)].unsqueeze(1)*(V[head_idx].T@model.model.decoder.layers[layer_idx].self_attn.q_proj.weight[80*(head_idx):80*(head_idx+1)]))
            model.model.decoder.layers[layer_idx].self_attn.q_proj.bias[80*(head_idx):80*(head_idx+1)].copy_(mask[80*(head_idx):80*(head_idx+1)]*(model.model.decoder.layers[layer_idx].self_attn.q_proj.bias[80*(head_idx):80*(head_idx+1)]@V[head_idx]))
            
            model.model.decoder.layers[layer_idx].self_attn.k_proj.weight[80*(head_idx):80*(head_idx+1)].copy_(mask[80*(head_idx):80*(head_idx+1)].unsqueeze(1)*(V[head_idx].T@model.model.decoder.layers[layer_idx].self_attn.k_proj.weight[80*(head_idx):80*(head_idx+1)]))
            model.model.decoder.layers[layer_idx].self_attn.k_proj.bias[80*(head_idx):80*(head_idx+1)].copy_(mask[80*(head_idx):80*(head_idx+1)]*(model.model.decoder.layers[layer_idx].self_attn.k_proj.bias[80*(head_idx):80*(head_idx+1)]@V[head_idx]))

        layerwise_pruning_ratio.append(1-mask.mean().item())

    print("Before Skewing")
    print(tokenizer.decode(outputs.logits.squeeze(0).argmax(dim=-1)))
    
    outputs = model(input_ids)
    print("After Skewing")
    print(tokenizer.decode(outputs.logits.squeeze(0).argmax(dim=-1)))
    
    key_values = outputs.past_key_values
    for layer_idx, (key, value) in enumerate(key_values):
        os.makedirs(f"graphs/keys/{layer_idx}", exist_ok=True)
        key = key.squeeze(0)
        for head_idx in range(key.size(0)):
            plt.imshow(key[head_idx].cpu().detach().abs(), cmap="Blues", norm=LogNorm())
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"graphs/keys/{layer_idx}/{head_idx}.png")
            plt.close()

plt.bar(range(len(layerwise_pruning_ratio)), layerwise_pruning_ratio)
plt.xlabel("Layer")
plt.ylabel("Pruning Ratio")
plt.savefig("Layerwise Pruning Ratio.png")

all_pruning_ratio = sum(layerwise_pruning_ratio)/len(layerwise_pruning_ratio)

model.save_pretrained(f"weights/skewed_opt-2.7b_{threshold}_{all_pruning_ratio:.3f}")