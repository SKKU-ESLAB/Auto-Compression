from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

torch.set_num_threads(4)
#FC = torch.load('../weight/fc3.pt').eval()
#m = 2048
#n = 4096

m = 50
n = 50
FC = nn.Linear(m,n).eval()

itr = 5

def push_flush():
    start = time.time() #####
    FC_F = torch.load('../weight/flush.pt').eval()
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    end = time.time()   #####

def mm_flush():
    fl.flush(FC.weight, m*n*4)

avg_time = 0
print("compute: fc layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m)
  
    #push_flush()
    #mm_flush()

    start = time.time() #####
    x = FC(x)
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
