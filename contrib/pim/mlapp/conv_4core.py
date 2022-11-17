from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("[CH,K1,K2]= 2,4,2:1, 4,6,3:2, 8,12,6:3, 16,24,12:4, 32,41,21:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

itr=5

CONV = torch.load('./weight/conv'+option+'.pt').eval()
#CONV = CONV.type(torch.bfloat16)
CONV = CONV.type(torch.int16)
avg_time = 0

print("compute: conv layer")
print("iter\t time")
for i in range(itr):
    #x = torch.randn((1, 1, 160, 1151))
    x = torch.randn((1, 1, 160, 1151)).type(torch.int16)
    
    start = time.time() #####
    #os.system('m5 dumpstats')
    FC_F = torch.load('./weight/flush.pt').eval()
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    #os.system('m5 dumpstats')
    end = time.time()   #####
    print("flush\t", end-start)

    start = time.time() #####
    os.system('m5 dumpstats')
    x = CONV(x)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
