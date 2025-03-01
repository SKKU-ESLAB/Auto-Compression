import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda:1"
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset("abisee/cnn_dailymail", "2.0.0")
ds_test = ds["test"]

special_tokens = list(tokenizer.special_tokens_map.values())
punctuation_tokens = [".", ",", "!", "?", ":", ";", "\"", "\'", "(", ")", "[", "]", "{", "}", "-", "_", "~", "*", "&"]
unicode = "0x"

def graph_variance(vec):
    if vec.dim() != 1:
        assert "graph_variance function is only for a 1D vector"
    tmp_vec = vec[vec!=-torch.inf]
    return (tmp_vec[:-2] + tmp_vec[2:] - 2*tmp_vec[1:-1]).abs().mean()

def make_vector(x):
    if len(x) == 0:
        return None
    
    result = torch.zeros_like(x[0])
    divider = torch.zeros_like(x[0])
    
    for tensor in x:
        tmp_tensor = tensor.clone()
        tmp_tensor[tmp_tensor == -torch.inf] = 0
        result += tmp_tensor
        divider += (tmp_tensor != 0.0).to(torch.float)
    
    return (result / divider).nan_to_num(nan=0.0)

def check_token(token, a):
    if isinstance(a, str):
        return a in token
    elif isinstance(a, list):
        return any(tmp in token for tmp in a)
    return False

def plot_attention(attentions, group_1, group_2, group_3, group_4, idx, layer, head):
    os.makedirs(f"tmp/prompt_{idx}/{layer}", exist_ok=True)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(attentions_list[0][layer, head].abs().pow(1/3), cmap="Blues")
    plt.xticks([]); plt.yticks([])
    plt.close()

    def plot_group(group, subplot_idx, title):
        if group:
            tmp_group = make_vector(group)
            if tmp_group is not None:
                # plt.subplot(1, 3, subplot_idx)
                # plt.title(f"{title} / Num: {len(group)}")
                for vector in group:
                    plt.title(graph_variance(vector))
                    # plt.title(vector[layer, head][vector[layer, head]!=-torch.inf].std())
                    plt.plot(vector[layer, head], color="darkred")#, alpha=0.01)
                    plt.savefig("tmp.png")
                    plt.close()
                    import pdb; pdb.set_trace()
                # plt.plot(tmp_group[layer, head])

    # plot_group(group_1, 2, "Special Tokens")
    plot_group(group_2, 2, "Punctuation Tokens")
    # plot_group(group_3, 4, "Unicode Tokens")
    plot_group(group_4, 3, "Normal Tokens")

    plt.tight_layout()
    plt.savefig(f"tmp/prompt_{idx}/{layer}/{head}.png")
    plt.close()

attentions_list = []

for idx in range(2,3,1):
    texts = ds_test[idx]["article"]
    inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False)
    tokenized_inputs = tokenizer.tokenize(texts)

    with torch.no_grad():
        outputs = model(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            output_attentions=True
        )

    attentions = torch.stack([attention.squeeze(0).cpu().to(torch.float) for attention in outputs.attentions])
    attentions_list.append(attentions)

max_seq_length = max(attention.size(-1) for attention in attentions_list)

for idx, attentions in enumerate(attentions_list):
    num_pad = max_seq_length - attentions.size(-1)
    attentions_list[idx] = F.pad(attentions, (0, num_pad, 0, num_pad))

# for token_id in [100, 200, 300, 400]:
#     for head_id in range(32):
#         plt.figure(figsize=(24,48))
#         for layer_id in range(32):
#             tmp_attentions = attentions[layer_id,head_id,:,token_id].roll(-token_id, -1)
#             plt.subplot(8,4,layer_id+1)
#             plt.title(f"Layer {layer_id}")
#             plt.plot(torch.log(tmp_attentions), label=layer_id)
#         plt.tight_layout()
#         os.makedirs(f"tmp/layer/{token_id}", exist_ok=True)
#         plt.savefig(f"tmp/layer/{token_id}/{head_id}.png")
#         plt.close()

# exit()

group_1, group_2, group_3, group_4 = [], [], [], []

for attentions in attentions_list:
    for token_id, token in enumerate(tokenized_inputs):
        tmp_attentions = attentions[:, :, :, token_id].roll(-token_id, -1)
        
        # Sink Token 
        if tmp_attentions[tmp_attentions!=0].mean() > 0.1:
            continue
        
        tmp_attentions = torch.log(tmp_attentions)
        
        if check_token(token, special_tokens):
            group_1.append(tmp_attentions)
        elif check_token(token, punctuation_tokens):
            group_2.append(tmp_attentions)
        elif check_token(token, unicode):
            group_3.append(tmp_attentions)
        else:
            group_4.append(tmp_attentions)

for layer in tqdm(range(32)):
    for head in range(32):
        plot_attention(attentions, group_1, group_2, group_3, group_4, idx, layer, head)
