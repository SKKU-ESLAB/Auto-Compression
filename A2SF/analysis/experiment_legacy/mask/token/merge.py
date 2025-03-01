from PIL import Image, ImageOps
import numpy as np

import os

file_path = os.path.dirname(__file__)

filenames = []

for idx in range(4):
    filenames.append(os.path.join(file_path, f"Token_{idx}.png"))

images = [Image.open(fname) for fname in filenames]

images = [img.resize(images[0].size) for img in images]

images = [ImageOps.expand(img, border=20, fill="white") for img in images]

np_images = [np.array(img) for img in images]

merged_image = np.hstack(np_images)

merged_image = Image.fromarray(merged_image)

merged_image.save(os.path.join(file_path, f"merge.png"))
