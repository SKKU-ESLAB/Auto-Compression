from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp

torch.set_num_threads(1)
LSTM = torch.load('../weight/lstm.pt').eval()

def push_flush():
    start = time.time() #####
    FC_F = torch.load('../weight/flush.pt').eval()
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    end = time.time()   #####
    #print("flush\t", end-start)

def mm_flush():
    flush(LSTM)
    #print("mm_flush~")

itr = 10
m = 576

print("compute: lstm layer")
avg_time = 0
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m, 1024)
    h0 = torch.randn(2, m, 512)
    c0 = torch.randn(2, m, 512)

    #push_flush()
    mm_flush()

    start = time.time() #####
    x, (h, c) = LSTM(x, (h0, c0))
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
