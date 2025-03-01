import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from transformers import AutoTokenizer

def a2s(vector):
    result = np.zeros_like(vector)
    
    for idx in range(vector.shape[-1]):
        result[idx] = np.sum(vector[0:idx+1])
    
    return result

def a2sf(vector, gamma = 0.2):
    result = np.zeros_like(vector)
    
    sum = 0
    for idx in range(vector.shape[-1]):
        sum += vector[idx]
        result[idx] = sum
        sum *= gamma

    return result

dir_path = os.path.dirname(__file__)

dataset = "arc_e"
layer = 15

prompts = {
    "winogrande": "Dharma wanted to bake some cookies and cakes for the bake sale. She ended up only baking the cakes because she didn't have a cookie sheet.\n\nI always wonder how people prefer reading in a library instead of at the house because the lack of people at the library would make it easier to concentrate.",
    "piqa": "Question: To make pumpkin spice granola\nAnswer: With a wooden spoon, mix together 2 1/2 tsp pumpkin pie spice, 1/4 tsp salt, 1/4 cup light brown sugar, 1/3 cup canned pumpkin puree (not pumpkin pie filling) , 2 Tbsp unsweetened applesauce, 2 Tbsp honey, 1/2 tsp vanilla extract, 1/4 cup raisins, 1/4 cup craisins in a medium bowl. Mix into 3 cups rolled oats until evenly coated. Line a cookie sheet with parchment paper. Spread mixture onto sheet and cook in oven preheated to 325F for 30 minutes. Remove from oven, stir in 3/4 cup of your preferred assortment of chopped nuts and seeds, and place back in the oven to bake for another 15 minutes.\n\nQuestion: how do you cheese something?\nAnswer: sprinkle cheese all over it.",
    "openbookqa": "A man is filling up his tank at the gas station, then goes inside and pays. When he comes back out, his truck is being stolen! He chases the truck as the thief pulls away in it and begins to speed off. Though the truck was huge, right in front of him a moment ago, as the truck accelerates and the man struggles to keep up, the truck looks smaller\n\nA positive effect of burning biofuel is shortage of crops for the food supply",
    "arc_e": "Question: Homes that are built to be environmentally friendly because they use energy more efficiently than other homes are called \"green\" homes. \"Green\" homes often have reflective roofs and walls made of recycled materials. The windows in these energy-saving homes are double-paned, meaning each window has two pieces of glass. Double-paned windows have a layer of air between the window panes. This layer is a barrier against extreme temperatures and saves energy. A solar panel on a \"green\" home uses\nAnswer: a renewable energy source\n\nQuestion: Which is the function of the gallbladder?\nAnswer: store bile",
    "mathqa": "Question: at company x , senior sales representatives visit the home office once every 15 days , and junior sales representatives visit the home office once every 10 days . the number of visits that a junior sales representative makes in a 2 - year period is approximately what percent greater than the number of visits that a senior representative makes in the same period ?\nAnswer: 50 %\n\nQuestion: a circle graph shows how the budget of a certain company was spent : 61 percent for salaries , 10 percent for research and development , 6 percent for utilities , 5 percent for equipment , 3 percent for supplies , and the remainder for transportation . if the area of each sector of the graph is proportional to the percent of the budget it represents , how many degrees of the circle are used to represent transportation ?\nAnswer:"
}

prompt = prompts[dataset]

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)

input_idx = tokenizer(prompt)

tensor = np.load(os.path.join(dir_path, dataset, "no_pruning", f"{layer}.npy"))
tensor = tensor[0][2] # batch, head

num_tokens = tensor.shape[0]
x_value = np.arange(num_tokens)

plt.figure(figsize=(16,4))

tmp = []
for idx in range(num_tokens):
    result = tensor[idx:,idx]
    tmp.append(np.sum(result))
tmp = np.array(tmp)
argm = np.argsort(tmp)[::-1]

for i in argm:
    print(f"{i} : {tokenizer.decode(input_idx.input_ids[i])} : {tmp[i]:.3f}")

for i, idx in enumerate([21,49,74,113]):
    result = tensor[idx:,idx]
    aas = a2s(result)
    aasf = a2sf(result, gamma=0.2)
    x = x_value[idx:]
    
    fig, ax1 = plt.subplots()
    
    token = tokenizer.decode(input_idx.input_ids[idx])
    plt.title(f"Token: {token}", fontsize=19)
    plt.xlabel("Generation Step", fontsize=17)
    plt.xlim((-1, num_tokens))
    
    if i == 0:
        plt.ylabel("Attention Score or A2SF", fontsize=17)
    ax1.plot(x, result, label="Attention Score")
    ax1.plot(x, aasf, label="A2SF")
    ax1.axvline(x=125, color='r', linestyle='--')
    ax1.set_ylim((0, 0.12))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    ax2 = ax1.twinx()
    if i == 3:
        ax2.set_ylabel("A2S", fontsize=17)
    ax2.plot(x, aas, label="A2S", c="tab:green")
    ax2.tick_params(axis="y", labelsize=17)
    ax2.set_ylim((0, 2.55))
    
    # fig.legend(bbox_to_anchor=(0.81, 0.85), ncol=3, fontsize=12)
    fig.tight_layout()

    plt.savefig(os.path.join(dir_path, "token", f"Token_{i}.png"))
    
    plt.close()

file_path = os.path.join(os.path.dirname(__file__), "token")

plt.bar(range(num_tokens), tmp)
plt.savefig(os.path.join(file_path, "test.png"))
plt.close()

filenames = []

for idx in range(4):
    filenames.append(os.path.join(file_path, f"Token_{idx}.png"))

images = [Image.open(fname) for fname in filenames]

images = [img.resize(images[0].size) for img in images]

images = [ImageOps.expand(img, border=0, fill="white") for img in images]

np_images = [np.array(img) for img in images]

merged_image = np.hstack(np_images)

merged_image = Image.fromarray(merged_image)

merged_image.save(os.path.join(file_path, f"merge.png"))