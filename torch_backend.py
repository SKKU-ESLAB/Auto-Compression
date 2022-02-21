import torch
import os

print("ha--->", end='')
os.system('m5 exit')
print("-->ha")

#print(*torch.__config__.show().split("\n"), sep="\n")
print(*torch.__config__.show().split("\n"), sep="\n")

print("\n*********************************************************\n")

print(*torch.__config__.parallel_info().split("\n"), sep="\n")
