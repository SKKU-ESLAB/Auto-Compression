import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from utils.original_llama import LlamaForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("abisee/cnn_dailymail", "2.0.0")
ds_test = ds["test"]

def make_vector(group):
    if not group:
        return None

    max_dim = max(tensor.size(2) for tensor in group)
    result = torch.zeros_like(group[0][:, :, :max_dim])
    divider = torch.zeros_like(result)

    for tensor in group:
        result[:, :, :tensor.size(2)] += tensor
        divider[:, :, :tensor.size(2)] += 1

    return result / divider

def plot_attention(group_weights, group_scores, layer, head):
    def plot_group(group, subplot_idx, title):
        tmp_group = make_vector(group)
        if tmp_group is not None:
            plt.subplot(2, 2, subplot_idx)
            plt.title(f"{title} / Num: {len(group)}")
            for vector in group:
                plt.plot(vector[layer, head], color="darkred", alpha=0.01)
            plt.plot(tmp_group[layer, head])

    os.makedirs(f"tmp/prompt_0/{layer}", exist_ok=True)
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    tmp_attn = weights_list[0][layer, head]
    tmp_attn = torch.max(tmp_attn, tmp_attn.tril(0).min())
    plt.imshow(tmp_attn, cmap="Blues")
    plt.xticks()
    plt.yticks()

    plot_group(group_weights, 2, "Attention Weights")
    
    plt.subplot(2, 2, 3)
    tmp_attn = scores_list[0][layer, head].pow(1/2)
    plt.imshow(tmp_attn, cmap="Blues")
    plt.xticks()
    plt.yticks()

    plot_group(group_scores, 4, "Punctuation Tokens")

    plt.tight_layout()
    plt.savefig(f"tmp/prompt_0/{layer}/{head}.png")
    plt.close()

weights_list = []
scores_list = []

for idx in range(2,3,1):
    texts = ds_test[idx]["article"]
    input_ids = tokenizer.encode(texts, return_tensors="pt", add_special_tokens=False)
    tokenized_inputs = tokenizer.tokenize(texts)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    weights_list.append(torch.stack([attention[0].squeeze(0) for attention in outputs.attentions]))
    scores_list.append(torch.stack([attention[1].squeeze(0) for attention in outputs.attentions]))

group_weights, group_scores = [], []

for weights, scores in zip(weights_list, scores_list):
    for token_id, token in enumerate(tokenized_inputs):
        group_weights.append(weights[:, :, token_id:, token_id])
        group_scores.append(torch.log(scores[:, :, token_id:, token_id] + 1e-10))
        import pdb; pdb.set_trace()

grid = []
for layer in range(32):
    for head in range(32):
        grid.append((layer, head))

for layer, head in tqdm(grid):
    plot_attention(group_weights, group_scores, layer, head)