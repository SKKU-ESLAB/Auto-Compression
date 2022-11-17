from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

BN1d = nn.BatchNorm1d(1024).eval()
BN1d.weight.require_grad = False
torch.save(BN1d, './weight/bn.pt')
