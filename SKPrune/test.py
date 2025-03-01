import torch
import matplotlib.pyplot as plt
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

model = "facebook/opt-2.7b"

config = AutoConfig.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model).eval()

layer_num = config.num_hidden_layers
head_num = config.num_attention_heads
head_dim = int(config.hidden_size/head_num)

dummy_text = "he Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed \"in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014.\" Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body."

with torch.no_grad():
    dummy_input = tokenizer(dummy_text, return_tensors="pt")

    original_output = model(input_ids=dummy_input.input_ids, attention_mask=dummy_input.attention_mask, output_attentions=True)
    original_kv_cache = original_output.past_key_values
    original_attentions = original_output.attentions

for model_name in ["weights/skewed_opt-2.7b_0.5_0.419"]:
    skewed_model = AutoModelForCausalLM.from_pretrained(model_name)

    with torch.no_grad():
        skewed_output = skewed_model(input_ids=dummy_input.input_ids, attention_mask=dummy_input.attention_mask, output_attentions=True)
        skewed_kv_cache = skewed_output.past_key_values
        skewed_attentions = skewed_output.attentions

    print("### original output ###")
    print(tokenizer.decode(original_output.logits[0].argmax(dim=-1)))
    print()
    print("### skewed output ###")
    print(tokenizer.decode(skewed_output.logits[0].argmax(dim=-1)))

    asdf = [(((original_attentions[i]*skewed_attentions[i]).sum())/(original_attentions[i].norm()*skewed_attentions[i].norm())).item() for i in range(32)]
    
    fs = 12
    plt.figure(figsize=(5,4))
    plt.plot(asdf)
    plt.title("Layerwise Attention Matrix Cosine Similarity", fontsize=fs+2)
    plt.xlabel("Layer Idx", fontsize=fs)
    plt.ylabel("Cosine Similarity", fontsize=fs)
    plt.ylim(0,1.1)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    plt.savefig("Layerwise_Attention_Matrix_Cosine_Similarity.png")
    plt.close()

    ratio = skewed_kv_cache[0][0].size(-1)/skewed_kv_cache[0][0].size(-2)

    for layer in tqdm(range(layer_num)):
        for head in range(head_num):
            original = original_kv_cache[layer][0][0,head].detach()
            skewed = skewed_kv_cache[layer][0][0,head].detach()
            
            fig, axs = plt.subplots(1, 2, figsize=(10*ratio, 5))
            
            axs[0].set_title("original key cache")
            axs[0].imshow(torch.abs(original), cmap="Blues", aspect="auto")
            axs[0].axis("off")
            axs[1].set_title("skewed key cache")
            axs[1].imshow(torch.abs(skewed), cmap="Blues", aspect="auto")
            axs[1].axis("off")
            
            plt.tight_layout()
            if not os.path.exists(f"graphs/{model_name}/keys/{layer}"):
                os.makedirs(f"graphs/{model_name}/keys/{layer}")
            plt.savefig(f"graphs/{model_name}/keys/{layer}/{head}.png")
            plt.close()

            original = original_kv_cache[layer][1][0,head].detach()
            skewed = skewed_kv_cache[layer][1][0,head].detach()
            
            fig, axs = plt.subplots(1, 2, figsize=(10*ratio, 5))
            
            axs[0].set_title("original value cache")
            axs[0].imshow(torch.abs(original), cmap="Blues", aspect="auto")
            axs[0].axis("off")
            axs[1].set_title("skewed value cache")
            axs[1].imshow(torch.abs(skewed), cmap="Blues", aspect="auto")
            axs[1].axis("off")
            
            plt.tight_layout()
            if not os.path.exists(f"graphs/{model_name}/values/{layer}"):
                os.makedirs(f"graphs/{model_name}/values/{layer}")
            plt.savefig(f"graphs/{model_name}/values/{layer}/{head}.png")
            plt.close()

            original = original_attentions[layer][0,head].detach()
            skewed = skewed_attentions[layer][0,head].detach()
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            axs[0].set_title("original attentions")
            axs[0].imshow(torch.pow(original, 1/3), cmap="Blues", aspect="auto")
            axs[0].axis("off")
            axs[1].set_title("skewed attentions")
            axs[1].imshow(torch.pow(skewed, 1/3), cmap="Blues", aspect="auto")
            axs[1].axis("off")
            
            plt.tight_layout()
            if not os.path.exists(f"graphs/{model_name}/attentions/{layer}"):
                os.makedirs(f"graphs/{model_name}/attentions/{layer}")
            plt.savefig(f"graphs/{model_name}/attentions/{layer}/{head}.png")
            plt.close()