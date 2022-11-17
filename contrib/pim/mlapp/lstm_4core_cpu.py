from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("# words = 1: 64, 2: 256, 3: 576\nenter layer to run: ")

print("option: ", option)

torch.set_num_threads(4)
print("\n----lets run!----")

m=0
itr=20

# lstm
if (option == '1'):
    m = 64
elif (option == '2'):
    m = 256
elif (option == '3'):
    m = 576

LSTM = torch.load('./weight/lstm.pt').eval()
#LSTM = LSTM.type(torch.int16)

print("compute: lstm layer")

avg_time = 0
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m, 1024)
    h0 = torch.randn(2, m, 512)
    c0 = torch.randn(2, m, 512)
    #x = torch.randn(1, m, 1024).type(torch.int16)
    #h0 = torch.randn(2, 64, 512).type(torch.int16)
    #c0 = torch.randn(2, 64, 512).type(torch.int16)

    start = time.time() #####
    FC_F = torch.load('./weight/flush.pt').eval()
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    end = time.time()   #####
    print("flush\t", end-start)

    start = time.time() #####
    x, (h, c) = LSTM(x, (h0, c0))
    #x = LSTM(x)
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
