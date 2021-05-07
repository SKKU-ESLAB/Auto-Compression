import os

os.system("python3 train.py app:apps/cifar10/0506_mobilenet_v2_lr0.01_k0.1.yml")
os.system("python3 train.py app:apps/cifar10/0506_mobilenet_v2_lr0.025_k0.1.yml")
os.system("python3 train.py app:apps/cifar10/0506_mobilenet_v2_lr0.05_k0.1.yml")
os.system("python3 train.py app:apps/cifar10/0506_mobilenet_v2_lr0.1_k0.1.yml")
