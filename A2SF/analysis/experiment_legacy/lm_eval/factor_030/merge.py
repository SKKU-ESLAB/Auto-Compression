from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os

file_path = os.path.dirname(__file__)

filenames = []

datasets = ["piqa", "copa", "arc_challenge", "mathqa"]
# models = ["LLaMA 2 7B", "LLaMA 7B", "OPT 6.7B", "OPT 2.7B"]
models = ["LLaMA 2 7B", "OPT 6.7B"]

for dataset in datasets:
    for model in models:
        filenames.append(os.path.join(file_path, "plot", dataset, f"{model} | 1-shot | {dataset}.png"))

images = [Image.open(fname) for fname in filenames]

images = [img.resize(images[0].size) for img in images]

np_images = [np.array(img) for img in images]

merged_image = np.hstack([np.vstack(np_images[i*len(models):(i+1)*len(models)]) for i in range(len(datasets))])

merged_image = Image.fromarray(merged_image)

merged_image.save(os.path.join(file_path, "Plot.png"))
