from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

option = input("# words = 1: 64, 2: 256, 3: 576\nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)
print("\n----lets run!----")

m=0
itr=5

# lstm
if (option == '0'):
    m = 1
elif (option == '1'):
    m = 64
elif (option == '2'):
    m = 256
elif (option == '3'):
    m = 576

LSTM = torch.load('./weight/lstm.pt').eval()

# FC_F = torch.load('./weight/flush.pt').eval()
avg_time = 0

print("compute: lstm layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m, 1024)
    h0 = torch.randn(2, m, 512)
    c0 = torch.randn(2, m, 512)

    start = time.time() #####
    os.system('m5 dumpstats')
    """
    in_F = torch.randn(2048)
    out_F = FC_F(in_F)
    """
    fl.flush(LSTM.weight_hh_l0, 512*4096*4)
    fl.flush(LSTM.weight_hh_l0_reverse, 512*4096*4)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print("flush\t", end-start)
    
    start = time.time() #####
    os.system('m5 dumpstats')
    x, (h, c) = LSTM(x, (h0, c0))
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
