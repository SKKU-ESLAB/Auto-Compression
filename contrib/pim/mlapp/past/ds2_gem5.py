from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 exit')
os.system('echo This is running on O3 CPU cores.')
print("\n----lets run!----")

# conv12
if (option == '1' or option == '5'):
    """
    layerCONV = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5)),
        nn.BatchNorm2d(32),
        nn.Hardtanh(0, 20, inplace=True),
        nn.Conv2d(32, 32, kernel_size=(21,11), stride=(2,1), padding=(10,5)),
        nn.BatchNorm2d(32),
        nn.Hardtanh(0, 20, inplace=True)
    )
    """
    layerCONV = torch.load('./weight/CONV')

# bi-lstm1
if (option == '2' or option == '5'):
    """
    layerLSTM1 = nn.Sequential(
        nn.LSTM(1280, 512, bidirectional=True, bias=True)
    )
    """
    layerLSTM1 = torch.load('./weight/LSTM1')


# bi-lstm23456
if (option == '3' or option == '5'):
    """
    layerBN1 = nn.BatchNorm1d(1024)
    layerLSTM2 = nn.LSTM(1024, 512, bidirectional=True, bias=True)
    """
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')

# fc1
if (option == '4' or option == '5'):
    """
    layerBN2 = nn.BatchNorm1d(1024)
    layerFC = nn.Linear(1024, 29)
    """
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')

if (option == '1'):
    print("compute: convolution layer 1 and 2")

    avg_time = 0
    print("iter\t time")
    for i in range(110):
        x = torch.randn((1, 1, 160, 1151))
        start = time.time() #####
        x = layerCONV(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)
        end = time.time()   #####
        print(i, "\t", end-start)
        if i >= 10:
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)

elif (option == '2'):
    print("compute: lstm layer 1")

    avg_time = 0
    print("iter\t time")
    for i in range(110):
        x = torch.randn((576, 1, 1280))
        start = time.time() #####
        x, _ = layerLSTM1(x)
        end = time.time()   #####
        print(i, "\t", end-start)
        if i >= 10:
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)

elif (option == '3'):
    print("compute: lstm layer 2 (or 3 4 5 6)")
    
    avg_time = 0
    print("iter\t time")
    for i in range(110):
        x = torch.randn((576, 1, 1024))
        start = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        end = time.time()   #####
        print(i, "\t", end-start)
        if i >= 10:
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)

elif (option == '4'):
    print("compute: fc layer")

    avg_time = 0
    print("iter\t time")
    for i in range(110):
        x = torch.randn((576, 1, 1024))
        start = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN2(x)
        x = x.view(sizes[0], sizes[1], -1)
        x = layerFC(x)
        end = time.time()   #####
        print(i, "\t", end-start)
        if i >= 10:
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)

elif (option == '5'):
    print("conpute: full ds2 model")

    avg_time = 0
    print("iter\t time")
    for i in range(110):
        x = torch.randn((1, 1, 160, 1151))
        start = time.time() #####
        x = layerCONV(x)
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
        if i >= 10:
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)
