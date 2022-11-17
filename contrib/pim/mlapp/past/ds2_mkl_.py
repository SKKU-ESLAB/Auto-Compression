from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
from torch.utils import mkldnn as mkldnn_utils

#torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5, all_in_one:6 \nenter layer to run: ")
print("option: ", option)
set_mkl = input("with_default:1, with_mkldnn:2\nenter backend to run: ")
print("set_mkl: ", set_mkl)

max_iter = 40
warm_iter = 5
num_iter = max_iter - warm_iter

# conv12
if (option == '1'):
    conv1 = torch.load('./weight/conv1')
    conv2 = torch.load('./weight/conv2')
    bn1 = torch.load('./weight/bn1')
    bn2 = torch.load('./weight/bn2')
    if (set_mkl == '2'):
        conv1 = mkldnn_utils.to_mkldnn(conv1)
        conv2 = mkldnn_utils.to_mkldnn(conv2)
        bn1 = mkldnn_utils.to_mkldnn(bn1)
        bn2 = mkldnn_utils.to_mkldnn(bn2)

# bi-lstm1
if (option == '2'):
    layerLSTM1 = torch.load('./weight/LSTM1')
    if (set_mkl == '2'):
        layerLSTM1 = mkldnn_utils.to_mkldnn(layerLSTM1)

# bi-lstm23456
if (option == '3'):
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')
    if (set_mkl == '2'):
        layerBN1 = mkldnn_utils.to_mkldnn(layerBN1)
        layerLSTM2 = mkldnn_utils.to_mkldnn(layerLSTM2)

# fc1
if (option == '4'):
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')
    if (set_mkl == '2'):
        layerBN2 = mkldnn_utils.to_mkldnn(layerBN2)
        layerFC = mkldnn_utils.to_mkldnn(layerFC)

# full model
if (option == '5' or option == '6'):
    conv1 = torch.load('./weight/conv1')
    conv2 = torch.load('./weight/conv2')
    bn1 = torch.load('./weight/bn1')
    bn2 = torch.load('./weight/bn2')
    layerLSTM1 = torch.load('./weight/LSTM1')
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')
    if (set_mkl == '2'):
        conv1 = mkldnn_utils.to_mkldnn(conv1)
        conv2 = mkldnn_utils.to_mkldnn(conv2)
        bn1 = mkldnn_utils.to_mkldnn(bn1)
        bn2 = mkldnn_utils.to_mkldnn(bn2)
        layerBN1 = mkldnn_utils.to_mkldnn(layerBN1)
        layerBN2 = mkldnn_utils.to_mkldnn(layerBN2)
        layerLSTM1 = mkldnn_utils.to_mkldnn(layerLSTM1)
        layerLSTM2 = mkldnn_utils.to_mkldnn(layerLSTM2)
        layerFC = mkldnn_utils.to_mkldnn(layerFC)
hardtanh = nn.Hardtanh(0, 20, inplace=True)
if (set_mkl == '2'):
    hardtanh = mkldnn_utils.to_mkldnn(hardtanh)

os.system('m5 exit')
os.system('echo CPU Switched!')
torch.set_num_threads(8)
print("\n----lets run!----")

def run_conv():
    print("compute: convolution layer 1 and 2")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))
        if (set_mkl == '2'):
            x = x.to_mkldnn()

        start = time.time() #####
        #x = hardtanh(bn2(conv2(hardtanh(bn1(conv1(x))))))
        x = conv1(x)
        #sizes = x.size()
        #x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        #x = x.transpose(1,2).transpose(0,1)
        end = time.time()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg_time: ", avg_time)
    return avg_time

def run_lstm1():
    print("compute: lstm layer 1")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1280))
        if (set_mkl == '2'):
            x = x.to_mkldnn()

        start = time.time() #####
        x, _ = layerLSTM1(x)
        end = time.time()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg_time: ", avg_time)
    return avg_time

def run_lstm2():
    print("compute: lstm layer 2 (or 3 4 5 6)")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1024))
        if (set_mkl == '2'):
            x = x.to_mkldnn()

        start = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        end = time.time()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg_time: ", avg_time)
    return avg_time

def run_fc():
    print("compute: fc layer")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1024))
        if (set_mkl == '2'):
            x = x.to_mkldnn()

        start = time.time() #####
        #sizes = x.size()
        #x = x.reshape(sizes[0]*sizes[1], -1)
        #x = layerBN2(x)
        #x = x.reshape(sizes[0], sizes[1], -1)
        x = layerFC(x)
        end = time.time()   #####
        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg_time: ", avg_time)
    return avg_time

def run_full_model():
    print("conpute: full ds2 model")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))
        if (set_mkl == '2'):
            x = x.to_mkldnn()

        start = time.time() #####
        x = hardtanh(bn2(conv2(hardtanh(bn1(conv1(x))))))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)

        x, _ = layerLSTM1(x)

        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)

        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)

        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN2(x)
        x = x.view(sizes[0], sizes[1], -1)
        x = layerFC(x)
        end = time.time()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
    avg_time = avg_time / num_iter
    print("avg_time: ", avg_time)
    return avg_time

if (option == '1'):
    run_conv()
elif (option == '2'):
    run_lstm1()
elif (option == '3'):
    run_lstm2()
elif (option == '4'):
    run_fc()
elif (option == '5'):
    run_full_model()
elif (option == '6'):
    conv_time = run_conv()
    lstm1_time = run_lstm1()
    lstm2_time = run_lstm2()
    fc_time = run_fc()
    full_model_time = run_full_model()

    print("\nTotal Result")
    print("conv layer:\t", conv_time)
    print("lstm1 layer:\t", lstm1_time)
    print("lstm2 layer:\t", lstm2_time)
    print("fc layer:\t", fc_time)
    print("full model:\t", full_model_time)

"""
layerCONV = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5)),
    nn.BatchNorm2d(32),
    nn.Hardtanh(0, 20, inplace=True),
    nn.Conv2d(32, 32, kernel_size=(21,11), stride=(2,1), padding=(10,5)),
    nn.BatchNorm2d(32),
    nn.Hardtanh(0, 20, inplace=True)
)
layerLSTM1 = nn.Sequential(
    nn.LSTM(1280, 512, bidirectional=True, bias=True)
)

layerBN1 = nn.BatchNorm1d(1024)
layerLSTM2 = nn.LSTM(1024, 512, bidirectional=True, bias=True)

layerBN2 = nn.BatchNorm1d(1024)
layerFC = nn.Linear(1024, 29)
"""
