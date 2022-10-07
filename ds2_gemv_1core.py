from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("5x10:1, 20x40:2, 80x160:3, 320x640:4, 1024x2048:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(1)
print("\n----lets run!----")

m=0
if (option == '1'):
    m = 5
elif (option == '2'):
    m = 20
elif (option == '3'):
    m = 80
elif (option == '4'):
    m = 320
elif (option == '5'):
    m = 1024
itr=5

FC = torch.load('./weight/fc'+option+'.pt').eval()
avg_time = 0
print("compute: fc layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn(1, m)
    
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
