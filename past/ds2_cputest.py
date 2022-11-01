from hwcounter import Timer, count, count_end
import torch
import torch.nn as nn
import time
import os

#torch.set_default_dtype(torch.bfloat16)
#torch.set_num_threads(4)
torch.set_grad_enabled(False)

option = input("save_fc:0, run_fc:1\nenter layer to run: ")

print("option: ", option)

max_iter = 40
warm_iter = 10
num_iter = max_iter - warm_iter

in_len = 1024
out_len = 512

# conv12
if (option == '0'):
    fc = nn.Linear(in_len,out_len).eval()
    fc.weight.requires_grad = False
    torch.save(fc, './weight/superlightfc')

if (option == '1'):
    fc = torch.load('./weight/superlightfc')

os.system('m5 exit')
os.system('echo CPU Switched!')
print("\n----lets run!----")

def run_fc():
    print("compute: fc layer")

    avg_clk = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn(in_len)

        start = count() #####
        x = fc(x)
        end = count()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_clk = avg_clk + end - start
    avg_clk = avg_clk / num_iter
    print("avg\t", avg_clk)
    return avg_clk

if (option == '1'):
    run_fc()
