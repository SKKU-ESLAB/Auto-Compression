import torch
import matplotlib.pyplot

def mode(x: list):
    tmp_dict = {}
    for i in set(x):
        tmp_dict[i] = 0
    for i in x:
        tmp_dict[i] += 1
    return tmp_dict

def partition(x):
    print(x[:,:,:10,:].mean(dim=(-1,-2)).argmax(dim=-1).to(torch.float).mean())
    print(x[:,:,10:22,:].mean(dim=(-1,-2)).argmax(dim=-1).to(torch.float).mean())
    print(x[:,:,22:,:].mean(dim=(-1,-2)).argmax(dim=-1).to(torch.float).mean())
    print()

openbookqa = torch.load("/home/smp9898/A2SF/analysis/forgetting_factor/openbookqa.pt")
piqa = torch.load("/home/smp9898/A2SF/analysis/forgetting_factor/piqa.pt")
arce = torch.load("/home/smp9898/A2SF/analysis/forgetting_factor/arc_easy.pt")
arcc = torch.load("/home/smp9898/A2SF/analysis/forgetting_factor/arc_challenge.pt")
mathqa = torch.load("/home/smp9898/A2SF/analysis/forgetting_factor/mathqa.pt")

import pdb; pdb.set_trace()