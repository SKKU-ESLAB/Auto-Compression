from unicodedata import bidirectional
import torch
import torch.nn as nn
import random
import time

#torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)
torch.set_num_threads(8)

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5 \nenter layer to run: ")

print("option: ", option)

# conv12
conv1 = nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5)).eval()
bn1 = nn.BatchNorm2d(32).eval()
conv2 = nn.Conv2d(32, 32, kernel_size=(21,11), stride=(2,1), padding=(10,5)).eval()
bn2 = nn.BatchNorm2d(32).eval()
conv1.weight.requires_grad = False
conv2.weight.requires_grad = False
bn1.weight.requires_grad = False
bn2.weight.requires_grad = False

layerCONV = nn.Sequential(
    conv1,
    bn1,
    nn.Hardtanh(0, 20, inplace=True),
    conv2,
    bn2,
    nn.Hardtanh(0, 20, inplace=True)
)

for i in range(32):
    bn1.weight[i] = random.random()
    bn2.weight[i] = random.random()

# bi-lstm1
layerLSTM1 = nn.LSTM(1280, 512, bidirectional=True, bias=True).eval()

# bi-lstm23456
layerBN1 = nn.BatchNorm1d(1024).eval()
layerLSTM2 = nn.LSTM(1024, 512, bidirectional=True, bias=True).eval()

# fc1
layerBN2 = nn.BatchNorm1d(1024).eval()
layerFC = nn.Linear(1024, 29).eval()

for i in range(1024):
    layerBN1.weight[i] = random.random()
    layerBN2.weight[i] = random.random()

for weights in layerLSTM1.all_weights:
    for weight in weights:
        weight.requires_grad = False

for weights in layerLSTM2.all_weights:
    for weight in weights:
        weight.requires_grad = False

layerBN1.weight.requires_grad = False
layerBN2.weight.requires_grad = False
layerFC.weight.requires_grad = False


if 1:
    torch.save(conv1, './weight/conv1')
    torch.save(bn1, './weight/bn1')
    torch.save(conv2, './weight/conv2')
    torch.save(bn2, './weight/bn2')
    torch.save(layerLSTM1, './weight/LSTM1')
    torch.save(layerLSTM2, './weight/LSTM2')
    torch.save(layerBN1, './weight/BN1')
    torch.save(layerBN2, './weight/BN2')
    torch.save(layerFC, './weight/FC')
else:
    layerCONV = torch.load('./weight/CONV')
    layerLSTM1 = torch.load('./weight/LSTM1')
    layerLSTM2 = torch.load('./weight/LSTM2')
    layerBN1 = torch.load('./weight/BN1')
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')


if 0:
    if (option == '1'):
        print("1!")

        avg_time = 0
        for i in range(110):
            x = torch.randn((1, 1, 160, 1151))
            start = time.time() #####
            x = layerCONV(x)
            sizes = x.size()
            x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
            x = x.transpose(1,2).transpose(0,1)
            end = time.time()   #####
            print(end-start)
            if i >= 10:
                avg_time = avg_time + end - start
        avg_time = avg_time / 100
        print("avg_time: ", avg_time)

    elif (option == '2'):
        print("2!")

        avg_time = 0
        for i in range(110):
            x = torch.randn((576, 1, 1280))
            start = time.time() #####
            x, _ = layerLSTM1(x)
            end = time.time()   #####
            print(end-start)
            if i >= 10:
                avg_time = avg_time + end - start
        avg_time = avg_time / 100
        print("avg_time: ", avg_time)

    elif (option == '3'):
        print("3!")
        
        avg_time = 0
        for i in range(110):
            x = torch.randn((576, 1, 1024))
            start = time.time() #####
            sizes = x.size()
            x = x.view(sizes[0]*sizes[1], -1)
            x = layerBN1(x)
            x = x.view(sizes[0], sizes[1], -1)
            x, _ = layerLSTM2(x)
            end = time.time()   #####
            print(end-start)
            if i >= 10:
                avg_time = avg_time + end - start
        avg_time = avg_time / 100
        print("avg_time: ", avg_time)

    elif (option == '4'):
        print("4!")

        avg_time = 0
        for i in range(110):
            x = torch.randn((576, 1, 1024))
            start = time.time() #####
            sizes = x.size()
            x = x.view(sizes[0]*sizes[1], -1)
            x = layerBN2(x)
            x = x.view(sizes[0], sizes[1], -1)
            x = layerFC(x)
            end = time.time()   #####
            print(end-start)
            if i >= 10:
                avg_time = avg_time + end - start
        avg_time = avg_time / 100
        print("avg_time: ", avg_time)

    elif (option == '5'):
        print("5!")

        avg_time = 0
        for i in range(110):
            x = torch.randn((1, 1, 160, 1151))
            x = layerCONV(x)
            sizes = x.size()
            x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
            x = x.transpose(1,2).transpose(0,1)

            x, _ = layerLSTM1(x)

            start = time.time() #####
            sizes = x.size()
            x = x.view(sizes[0]*sizes[1], -1)
            x = layerBN1(x)
            x = x.view(sizes[0], sizes[1], -1)
            x, _ = layerLSTM2(x)
            end = time.time()   #####

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

            print(end-start)
            if i >= 10:
                avg_time = avg_time + end - start
        avg_time = avg_time / 100
        print("avg_time: ", avg_time)

