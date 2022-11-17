from benchmarker import Benchmarker
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
    fc = nn.Linear(in_len, out_len).eval()
    fc.weight.requires_grad = False
    torch.save(fc, './weight/superlightfc')

if (option != '0'):
    fc = torch.load('./weight/superlightfc')

os.system('m5 exit')
os.system('echo CPU Switched!')
print("\n----lets run!----")

def run_fc():
    print("compute: fc layer")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn(in_len)

        start = time.time() #####
        x = fc(x)
        end = time.time()   #####

        print(i, "\t%.6f" %(end-start))
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg\t%.6f" %avg_time)
    return avg_time

if (option == '1'):
    run_fc()

with Benchmarker(1, width=20) as bench:
    @bench("haha")
    def _(bm):
        for i in bm:
            run_fc()
    
    @bench("baba")
    def _(bm):
        for i in bm:
            run_fc()
    
    @bench("caca")
    def _(bm):
        for i in bm:
            run_fc()
    
    @bench("siba")
    def _(bm):
        for i in bm:
            run_fc()