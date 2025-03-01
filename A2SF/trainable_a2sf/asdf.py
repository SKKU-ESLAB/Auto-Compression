import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model).eval().half().to("cuda")

prompt = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nSummarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."

input = tokenizer(prompt, return_tensors="pt")
input_ids = input.input_ids.to("cuda")
attention_mask = input.attention_mask.to("cuda")

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)

x = range(input_ids.size(-1))

for i in range(32):
    i_attnetion = output.attentions[i].sum(dim=-2).squeeze(dim=0).cpu().detach()
    for j in range(32):
        plt.bar(x, i_attnetion[j])
        plt.yscale("log", base=10)
        plt.savefig(f"tmp/{i}_{j}.png")
        plt.close()