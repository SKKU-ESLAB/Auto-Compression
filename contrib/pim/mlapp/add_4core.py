from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

option = input("1: 65536 (65K), 2: 131072 (131K), 3: 4194304 (4M)\nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

m=0
itr=5
if (option == '1'):
    m = 65536
elif (option == '2'):
    m = 131072
elif (option == '3'):
    m = 4194304

# FC_F = torch.load('./weight/flush.pt').eval()
avg_time = 0

print("compute: add")
print("iter\t time")
for i in range(itr):
    in0 = torch.randn(1, m)
    in1 = torch.randn(1, m)

    start = time.time() #####
    os.system('m5 dumpstats')
    """
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    """
    fl.flush(in0, m*4)
    fl.flush(in1, m*4)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print("flush\t", end-start)

    start = time.time() #####
    os.system('m5 dumpstats')
    out = in0 + in1;
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
