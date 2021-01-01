import os

exp = ['test-009','test-010','test-011','test-012']#,'test-013','test-014','test-015','test-008']
comp_ratio = 5.*8./32/32
w_bit = "2 4 6 8"
scaling = 1e-7
lr_ms = "20 40 60"
lr_g = 0.1
seed = [3, 5, 10, 22]
space = "".join(["\n"]*2)
print(space)
for i in range(len(exp)):
    string = f"CUDA_VISIBLE_DEVICES={i%2}  python3 train.py --exp {exp[i]} --epoch 80 --seed {seed[i]} "\
             f"--batchsize 128 --w_bit {w_bit} --x_bit 8 "\
             f"--scaling {scaling} --comp_ratio {comp_ratio} "\
             f"--lr_ms {lr_ms} --lr_g {lr_g} "\
              "-g -q -lb -lq --load 2 --load_file ./checkpoint/resnet32_c100_2.pth --strict_false"
    print(string)
    print(space)



'''
exp = ["A001", "A002", "A003"]
comp_ratio = [0.03125, 0.0390625, 0.046875]
w_bit = ["2 4 6 8"] * 3
scaling = [1e+06] * 3

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 --scaling {scaling[i]} --comp_ratio {comp_ratio[i]} -g -q -lb -lq"
    print(string)
    os.system(string)
'''
'''
exp = ["B001", "B002", "B003"]
comp_ratio = [0.03125, 0.0390625, 0.046875]
w_bit = ["2 4 6 8"]*3
scaling = [1e+06] * 3

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 --scaling {scaling[i]} --comp_ratio {comp_ratio[i]} -q -lb"
    print(string)
    os.system(string)
'''
'''
exp = ["C001", "C002", "C003"]
w_bit = [4, 5, 6]

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 -q -lq"
    print(string)
    os.system(string)

'''
'''

exp = ["D001", "D002", "D003"]
w_bit = [4, 5, 6]

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 -q"
    print(string)
    os.system(string)

'''
'''

exp = ["A007", "A008", "A009", "A010", "A011"]
comp_ratio = [0.0234375, 0.03125, 0.0390625, 0.046875, 0.0546875]
w_bit = ["2 3 4 5 6 7 8"] * 5
scaling = [1e+06] * 5

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 --scaling {scaling[i]} --comp_ratio {comp_ratio[i]} -g -q -lb -lq"
    print(string)
    os.system(string)

'''
'''

exp = ["B007", "B008", "B009", "B010", "B011"]
w_bit = [3, 4, 5, 6, 7]

for i in range(len(exp)):
    string = f"python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128 --w_bit {w_bit[i]} --x_bit 8 -q"
    print(string)
    os.system(string)

exp = ["Full"]

for i in range(len(exp)):
    string = f'python3 train.py --exp {exp[i]} --epoch 160 --batchsize 128'
    print(string)
    os.system(string)

'''




#string = 'CUDA_VISIBLE_DEVICES=1 python3 train.py --load 1 --file_name save/1107_pre_01_best_ckpt.pth --exp 1111_07_FWFW -fwbw -fwlq -lq -q -g --batchsize 100 --s_ms 20 40 60 80 --s_g 0.2 --lr1 0.01 --lr2 0.001  --lr3 0.0001 --w_bit 3 --x_bit 4'
#string = 'CUDA_VISIBLE_DEVICES=1 python3 train.py --load 1 --file_name save/1107_pre_01_best_ckpt.pth --exp 1110_08_FWLW -fwbw -lq -q -g --batchsize 128 --s_ms 20 40 60 80 --s_g 0.2 --lr1 0.01 --lr2 0.001  --lr3 0.0001 --x_bit 4'
#string = 'CUDA_VISIBLE_DEVICES=1 python3 train.py --load 1 --file_name save/1107_pre_01_best_ckpt.pth --exp 1109_12_FbwFW -lq -q -g -fwbw -fwlq --batchsize 256 --s_ms 20 40 60 --s_g 0.1 --lr1 0.01 --lr2 0.001  --lr3 0.0001'
#string = 'CUDA_VISIBLE_DEVICES=1 python3 train.py --load 1 --file_name save/1107_pre_01_best_ckpt.pth --exp 1109_14_FbwFW -lq -q -g -fwbw -fwlq --batchsize 1024 --s_ms 20 40 60 --s_g 0.1 --lr1 0.01 --lr2 0.001  --lr3 0.0001'
