import torch
import time
import os
from torch.utils import mkldnn as mkldnn_utils

option = input("default:0, mkl:1\nenter option to run: ")

print("option: ", option)

max_iter = 20000
avg_time = 0

fc = torch.nn.Linear(1024,512)
x = torch.rand(16,1024)

os.system('m5 exit')
os.system('echo CPU Switched!')
#torch.set_num_threads(4)

if int(option) == 1:
    print(fc)
    fc = mkldnn_utils.to_mkldnn(fc)
    print(fc)
    print(x)
    x = x.to_mkldnn()
    print(x)

for i in range(max_iter):
    start = time.time()
    y = fc(x)
    end = time.time()

    avg_time = avg_time + end - start
avg_time = avg_time / max_iter

print("avg_time: %.6f" %avg_time)
