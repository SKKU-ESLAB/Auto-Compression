import os

a = 0

if a == 0:
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_fracbits.yml")
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_fracbits_BR.yml")
elif a == 1:
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_ws4_v2_L1_LD.yml")
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_ws4_v2_L1_LD_BR.yml")
elif a == 2:
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_ws4_v2_L2.yml")
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_ws4_v2_L2_LD.yml")
else:
    os.system("python3 train.py app:apps/cifar10/0510_rsn20_ws4_v2_L2_LD_BR.yml")

"""
if a == 0:
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_L1_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_L2_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_L3_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_L4_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_L_learned_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_sm_tau0.5_lr0.05.yml")
elif a == 1:
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v2_L1_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v2_L2_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v2_L3_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v2_L4_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_sm_tau1_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v1_sm_tau2_lr0.05.yml")
elif a == 2:
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v1_L1_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v1_L2_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v1_L3_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v1_L4_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws2_v2_sm_tau0.5_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v1_L_learned_lr0.05.yml")
else:
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v2_L4_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v2_L1_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v2_L2_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v2_L3_lr0.05.yml")
    os.system("python3 train.py app:apps/cifar10/0507_rsn20_ws4_v2_L_learned_lr0.05.yml")
"""
