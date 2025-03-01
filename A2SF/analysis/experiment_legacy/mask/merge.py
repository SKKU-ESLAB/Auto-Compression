from PIL import Image, ImageOps
import numpy as np

import os

file_path = os.path.dirname(__file__)

dataset = "winogrande"
layers = ["1", "30"]
methods = ["ideal", "local", "h2o", "a2sf_010", "a2sf_050"]

if not os.path.exists(os.path.join(file_path, "Plots")):
    os.makedirs(os.path.join(file_path, "Plots"))

for head in range(32):
    filenames = []

    for method in methods:
        for layer in layers:
            filenames.append(os.path.join(file_path, dataset, "mask", layer, method, f"test_{head}.png"))

    images = [Image.open(fname) for fname in filenames]

    images = [img.resize(images[0].size) for img in images]

    images = [ImageOps.expand(img, border=20, fill="white") for img in images]

    np_images = [np.array(img) for img in images]

    merged_image = np.hstack([np.vstack(np_images[i*len(layers):(i+1)*len(layers)]) for i in range(len(methods))])

    merged_image = Image.fromarray(merged_image)

    merged_image.save(os.path.join(file_path, "Plots", f"head_{head}.png"))
