import os
import numpy as np

from transformers import AutoTokenizer

import torch

dir_path = os.path.dirname(__file__)

prompts = {
    "winogrande": "Dharma wanted to bake some cookies and cakes for the bake sale. She ended up only baking the cakes because she didn't have a cookie sheet.\n\nI always wonder how people prefer reading in a library instead of at the house because the lack of people at the library would make it easier to concentrate.",
    "piqa": "Question: To make pumpkin spice granola\nAnswer: With a wooden spoon, mix together 2 1/2 tsp pumpkin pie spice, 1/4 tsp salt, 1/4 cup light brown sugar, 1/3 cup canned pumpkin puree (not pumpkin pie filling) , 2 Tbsp unsweetened applesauce, 2 Tbsp honey, 1/2 tsp vanilla extract, 1/4 cup raisins, 1/4 cup craisins in a medium bowl. Mix into 3 cups rolled oats until evenly coated. Line a cookie sheet with parchment paper. Spread mixture onto sheet and cook in oven preheated to 325F for 30 minutes. Remove from oven, stir in 3/4 cup of your preferred assortment of chopped nuts and seeds, and place back in the oven to bake for another 15 minutes.\n\nQuestion: how do you cheese something?\nAnswer: sprinkle cheese all over it.",
    "openbookqa": "A man is filling up his tank at the gas station, then goes inside and pays. When he comes back out, his truck is being stolen! He chases the truck as the thief pulls away in it and begins to speed off. Though the truck was huge, right in front of him a moment ago, as the truck accelerates and the man struggles to keep up, the truck looks smaller\n\nA positive effect of burning biofuel is shortage of crops for the food supply",
    "arc_e": "Question: Homes that are built to be environmentally friendly because they use energy more efficiently than other homes are called \"green\" homes. \"Green\" homes often have reflective roofs and walls made of recycled materials. The windows in these energy-saving homes are double-paned, meaning each window has two pieces of glass. Double-paned windows have a layer of air between the window panes. This layer is a barrier against extreme temperatures and saves energy. A solar panel on a \"green\" home uses\nAnswer: a renewable energy source\n\nQuestion: Which is the function of the gallbladder?\nAnswer: store bile",
    "mathqa": "Question: at company x , senior sales representatives visit the home office once every 15 days , and junior sales representatives visit the home office once every 10 days . the number of visits that a junior sales representative makes in a 2 - year period is approximately what percent greater than the number of visits that a senior representative makes in the same period ?\nAnswer: 50 %\n\nQuestion: a circle graph shows how the budget of a certain company was spent : 61 percent for salaries , 10 percent for research and development , 6 percent for utilities , 5 percent for equipment , 3 percent for supplies , and the remainder for transportation . if the area of each sector of the graph is proportional to the percent of the budget it represents , how many degrees of the circle are used to represent transportation ?\nAnswer:"
}

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

mask_list = {
    "IDEAL": [],
    "H2O": [],
    "A2SF_ZERO": [],
}

dataset = "mathqa"
layer = 15
head = 0

prompt = prompts[dataset]
input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids
tokens = tokenizer.tokenize(prompt)
tokens = np.insert(tokens, 0, "<s>")

attention_path = os.path.join(dir_path, "npy", dataset, "A2SF_ZERO", f"{layer}.npy")
attention_map = np.load(attention_path)[0, head] > 0

for i in range(attention_map.shape[0]):
    tmp = np.where(attention_map[i])[0]
    print(f"{[tokens[j] for j in tmp]}")