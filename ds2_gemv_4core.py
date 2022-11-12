from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("1: 512x4096, 2: 1024x4096, 3: 2048x4096\nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

m=0
n=4096
itr=5
if (option == '1'):
    m = 512
elif (option == '2'):
    m = 1024
elif (option == '3'):
    m = 2048

FC = torch.load('./weight/fc'+option+'.pt').eval()
FC = FC.type(torch.bfloat16)
avg_time = 0

print("compute: fc layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m).type(torch.bfloat16)
    
    start = time.time() #####
    os.system('m5 dumpstats')
    FC_F = torch.load('./weight/flush.pt').eval()
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print("flush\t", end-start)

    start = time.time() #####
    os.system('m5 dumpstats')
    x = FC(x)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
