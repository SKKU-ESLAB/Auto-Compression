from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("[CH,K1,K2]= 2,4,2:1, 4,6,3:2, 8,12,6:3, 16,24,12:4, 32,41,21:5 \nenter layer to run: ")

print("option: ", option)

print("\n----lets run!----")

m=0
n=0
itr=5

# fc1 
if (option == '1'):
    m = 2
    k1 = 4
    k2 = 2
 
elif (option == '2'):
    m = 4
    k1 = 6
    k2 = 3
 
elif (option == '3'):
    m = 8
    k1 = 12
    k2 = 6
  
elif (option == '4'):
    m = 16
    k1 = 24
    k2 = 12
 
elif (option == '5'):
    m = 32
    k1 = 41
    k2 = 21
    
conv1 = nn.Conv2d(1, m, kernel_size=(k1,11), stride=(2,2), padding=(20,5)).eval()
conv2 = nn.Conv2d(m, m, kernel_size=(k2,11), stride=(2,1), padding=(10,5)).eval()
conv1.weight.requires_grad = False
conv2.weight.requires_grad = False
CONV = nn.Sequential(conv1, conv2)

torch.save(CONV, './weight/conv'+option+'.pt')
