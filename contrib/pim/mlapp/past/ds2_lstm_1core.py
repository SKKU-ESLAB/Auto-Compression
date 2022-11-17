from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("# words = 32:1, 64:2, 128:3, 256:4, 576:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(1)
print("\n----lets run!----")

m=0
itr=5

# lstm
if (option == '1'):
    m = 32

elif (option == '2'):
    m = 64
 
elif (option == '3'):
    m = 128
 
elif (option == '4'):
    m = 256
 
elif (option == '5'):
    m = 576
    
LSTM = torch.load('./weight/lstm.pt').eval()

print("compute: lstm layer")

avg_time = 0
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m, 1024)
    
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
    x = LSTM(x)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)
