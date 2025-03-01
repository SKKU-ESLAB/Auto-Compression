import re
import os
import math
import matplotlib.pyplot as plt

dir_path = os.path.dirname(__file__)

fewshots = [1, 0]

models = [
    "LLaMA-2 7B",
    "LLaMA 7B"
]

cache_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

methods = [
    "IDEAL",
    "IDEAL_VALUE",
    "LOCAL",#
    "H2O",
    "A2SF_ZERO",
    "A2SF_RECENT",#
    "A2SF_TW_ZERO",#
    "A2SF_TW_RECENT",#
    "NOHIS_ZERO",
    "NOHIS_RECENT",#
    "A2SF_TENDANCY_ZERO",#
    "A2SF_TENDANCY_RECENT"
]

except_method = [
    "LOCAL",
    "A2SF_RECENT",
    "A2SF_TW_ZERO",
    "A2SF_TW_RECENT",
    "NOHIS_RECENT",
    "A2SF_TENDANCY_ZERO",
]

datasets = [
    "openbookqa",
    "winogrande",
    "piqa",
    "arc_easy",
    "arc_challenge"
]

result_dict = {}

for fewshot in fewshots:
    result_dict[fewshot] = {}
    for model in models:
        result_dict[fewshot][model] = {}
        for dataset in datasets:
            result_dict[fewshot][model][dataset] = {}
            result_dict[fewshot][model][dataset]["FULL"] = []
            for method in methods:
                result_dict[fewshot][model][dataset][method] = []
                

with open(os.path.join(dir_path, "result.txt"), "r") as f:
    file = f.readlines()

data = list()

for i in file:
    if "|acc_norm" not in i:
        if "|acc" in i:
            values = re.findall(r"\|acc\s*\|\s*(\d+\.\d+)", i)
            data.append(float(values[0]))

idx = 0
for fewshot in fewshots:
    for model in models:
        for dataset in datasets:
            result_dict[fewshot][model][dataset]["FULL"].append(data[idx])
            idx += 1
        for cache_ratio in cache_ratios:
            for method in methods:
                for dataset in datasets:
                    result_dict[fewshot][model][dataset][method].append(data[idx])
                    idx += 1

folder_path = os.path.join(dir_path, "plot")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

column = 3
row = math.ceil(len(datasets)/3)

for fewshot in fewshots:
    for model in models:
        idx = 1
        plt.figure(figsize=(5*column,5*row))
        for dataset in datasets:
            plt.subplot(2, 3, idx)
            plt.title(dataset, fontsize=20)
            plt.grid(True, linestyle="dashed")
            plt.gca().invert_xaxis()
            plt.plot(cache_ratios, result_dict[fewshot][model][dataset]["FULL"]*len(cache_ratios), linestyle="dashed", label="FULL")
            for method in [i for i in methods if i not in except_method]:
                plt.plot(cache_ratios, result_dict[fewshot][model][dataset][method], label=method, marker="o", markersize=5)
                
            # plt.legend(loc="lower left", fontsize=13)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.xlabel("Cache Ratio", fontsize=13)
            plt.ylabel("Accuracy", fontsize=13)
            plt.tight_layout()
            
            idx += 1
        
        plt.subplot(2, 3, idx)
        plt.plot(0, 0, linestyle="dashed", label="FULL")
        for method in [i for i in methods if i not in except_method]:
            plt.plot(0, 0, label=method, marker="o", markersize=5)
        plt.legend(loc="center", fontsize=13, framealpha=1.0)
        plt.xticks([])
        plt.yticks([])
        for side in ["top", "bottom", "left", "right"]:
            plt.gca().spines[side].set_visible(False)
        
        plt.savefig(os.path.join(folder_path, f"{fewshot}_{model}.png"))
        plt.close()
                