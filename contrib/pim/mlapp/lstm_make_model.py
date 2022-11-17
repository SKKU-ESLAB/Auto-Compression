from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

# lstm
LSTM = nn.LSTM(1024, 512, bidirectional=True, bias=True).eval()
torch.save(LSTM, './weight/lstm.pt')
