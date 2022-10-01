from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("# words = 32:1, 64:2, 128:3, 256:4, 576:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

# lstm
LSTM = nn.LSTM(1024, 512, bidirectional=True, bias=True).eval()
torch.save(LSTM, './weight/lstm.pt')
