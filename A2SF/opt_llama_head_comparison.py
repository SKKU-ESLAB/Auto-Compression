import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading OPT model...")
opt = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16)
print("OPT model loaded.")

print("Loading Llama model...")
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
print("Llama model loaded.")

print("Loading tokenizers...")
token_opt = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
token_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
print("Tokenizers loaded.")

prompt = json.loads(open("data/xsum-3shot.jsonl").readline())["article"]
input_ids_opt = token_opt.encode(prompt, return_tensors="pt")
input_ids_llama = token_llama.encode(prompt, return_tensors="pt")

print("Generating output for OPT model...")
output_opt = opt(input_ids_opt, output_attentions=True)
print("OPT model output generated.")

print("Generating output for Llama model...")
output_llama = llama(input_ids_llama, output_attentions=True)
print("Llama model output generated.")

attentions_opt = output_opt.attentions
attentions_llama = output_llama.attentions

print("Starting attention map visualization...")

for layer_num in range(len(attentions_opt)):  # 레이어 수
    for head_num in range(attentions_opt[layer_num].shape[1]):  # 헤드 수
        # subplot 생성
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # OPT 모델의 attention map 시각화
        attention_opt = attentions_opt[layer_num][0, head_num].detach().cpu().numpy()
        ax_opt = axes[0]
        im_opt = ax_opt.imshow(np.power(attention_opt, 1/3), cmap='Blues')
        ax_opt.set_title(f"OPT Layer {layer_num} Head {head_num}")

        # Llama 모델의 attention map 시각화
        attention_llama = attentions_llama[layer_num][0, head_num].detach().cpu().numpy()
        ax_llama = axes[1]
        im_llama = ax_llama.imshow(np.power(attention_llama, 1/3), cmap='Blues')
        ax_llama.set_title(f"Llama Layer {layer_num} Head {head_num}")

        # 파일 경로 생성 및 저장
        os.makedirs(f"tmp/{layer_num}", exist_ok=True)
        plt.savefig(f"tmp/{layer_num}/head_{head_num}_comparison.png")
        plt.close()

        print(f"Saved attention map for Layer {layer_num}, Head {head_num}")

print("Attention maps saved successfully.")
