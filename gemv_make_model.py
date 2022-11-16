from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("1: 512x4096, 2: 1024x4096, 3: 2048x4096\nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
print("\n----lets run!----")

m=0
n=0

# fc1
if (option == '1'):
    m = 512
    n = 4096

elif (option == '2'):
    m = 1024
    n = 4096
 
elif (option == '3'):
    m = 2048
    n = 4096
 
FC = nn.Linear(m, n).eval()
FC.weight.require_grad = False
torch.save(FC, './weight/fc'+option+'.pt')

