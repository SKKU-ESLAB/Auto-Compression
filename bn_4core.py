from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

option = input("1: 64-1024, 2: 256-1024, 3: 576-1024\nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

l=0
f=1024
itr=5
if (option == '1'):
    l = 64
elif (option == '2'):
    l = 256
elif (option == '3'):
    l = 576

BN1d = torch.load('./weight/bn.pt').eval()

# FC_F = torch.load('./weight/flush.pt').eval()
avg_time = 0

print("compute: bn1d layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn(l, f)

    start = time.time() #####
    os.system('m5 dumpstats')
    """
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    """
    fl.flush(x, l*f*4)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print("flush\t", end-start)

    start = time.time() #####
    os.system('m5 dumpstats')
    x = BN1d(x)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
