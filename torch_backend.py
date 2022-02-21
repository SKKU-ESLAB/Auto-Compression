import torch
import os

print("ha--->", end='')
os.system('m5 exit')
print("-->ha")

#torch.set_num_threads(4)

print(*torch.__config__.show().split("\n"), sep="\n")

print("\n*********************************************************\n")

print(*torch.__config__.parallel_info().split("\n"), sep="\n")
