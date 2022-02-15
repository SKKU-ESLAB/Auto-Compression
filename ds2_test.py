import torch
import torch.nn as nn
import time
import os

#torch.set_default_dtype(torch.bfloat16)
#torch.set_num_threads(8)
torch.set_grad_enabled(False)

option = input("save_fc:0, run_fc:1\nenter layer to run: ")

print("option: ", option)

max_iter = 10
warm_iter = 2
num_iter = max_iter - warm_iter

# conv12
if (option == '0'):
    fc = nn.Linear(1024,512).eval()
    fc.weight.requires_grad = False
    torch.save(fc, './weight/superlightfc')

if (option == '1'):
    fc = torch.load('./weight/superlightfc')

os.system('m5 exit')
os.system('echo CPU Switched!')
print("\n----lets run!----")

def run_fc():
    print("compute: fc layer")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn(1024)

        start = time.time() #####
        x = fc(x)
        end = time.time()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg\t", avg_time)
    return avg_time

if (option == '1'):
    run_fc()
