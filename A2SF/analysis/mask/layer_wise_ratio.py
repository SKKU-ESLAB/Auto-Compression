import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

def make_mask(input_tensor, heavy_ratio, penalty):
    tensor = torch.clone(input_tensor)
    cache_budget = int(tensor.shape[-2]*heavy_ratio)
    
    a2sf = torch.zeros_like(tensor[:,0,:])
    tmp_mask = torch.ones_like(tensor[:,0,:])
    
    for i in range(cache_budget):
        a2sf = penalty*a2sf + tensor[:,i,:]
    
    for i in range(cache_budget, tensor.shape[-2]):
        current_score = tensor[:,i,:]
        
        current_score *= tmp_mask
        current_score /= torch.sum(current_score, axis=-1).unsqueeze(dim=-1)
        
        if i != tensor.shape[-2]-1:
            if penalty != 0.0:
                a2sf = penalty*a2sf + current_score
            else:
                a2sf[a2sf!=torch.inf] = 0
                a2sf += current_score
        
            min_index = torch.argmin(a2sf[:,:i+1], axis=-1).unsqueeze(dim=-1)
            tmp_mask.scatter_(-1, min_index, 0)
            a2sf.scatter_(-1, min_index, np.inf)

    return tensor

def similarity(tensor_a, tensor_b):
    return torch.sum(torch.multiply(tensor_a, tensor_b))/(torch.norm(tensor_a)*torch.norm(tensor_b) + 1e-10)

dir_path = os.path.dirname(__file__)

penalty = 0.1

for dataset in ["mathqa", "arc_e", "winogrande", "piqa", "openbookqa"]:
    tensors = []
    for i in range(32):
        tensors.append(np.load(os.path.join(dir_path, "npy", dataset, "NO_PRUNING", f"{i}.npy")))
    tensors = np.concatenate(tensors, axis=0)
    tensors = torch.from_numpy(tensors)

    ratios = 0.2*torch.ones(tensors.shape[0])
    similarities = torch.tensor([similarity(tensors[i], make_mask(tensors[i], ratios[i], penalty)) for i in range(ratios.shape[0])])

    for i in tqdm(range(100)):
        max_i = torch.argmax(similarities)
        min_i = torch.argmin(similarities)
        
        ratios[max_i] -= 0.01
        ratios[min_i] += 0.01
        
        similarities[max_i] = similarity(tensors[max_i], make_mask(tensors[max_i], ratios[max_i], penalty))
        similarities[min_i] = similarity(tensors[min_i], make_mask(tensors[min_i], ratios[min_i], penalty))

    print(dataset)
    for i in range(ratios.shape[0]): print(f"{ratios[i]:.2f}")

    print(ratios)