from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

FC_f = nn.Linear(2048, 2048).eval()
FC_f.weight.requiure_grad = False
torch.save(FC_f, './weight/flush.pt')
