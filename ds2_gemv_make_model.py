from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("5x10:1, 20x40:2, 80x160:3, 320x640:4, 1024x2048:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
print("\n----lets run!----")

m=0
n=0
itr=5

# fc1
if (option == '1'):
    m = 5
    n = 10

elif (option == '2'):
    m = 20
    n = 40
 
elif (option == '3'):
    m = 80
    n = 160
 
elif (option == '4'):
    m = 320
    n = 640
 
elif (option == '5'):
    m = 1024
    n = 2048
    
FC = nn.Linear(m, n).eval()
FC.weight.requiure_grad = False
torch.save(FC, './weight/fc'+option+'.pt')
