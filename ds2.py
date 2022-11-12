from unicodedata import bidirectional
import torch
import torch.nn as nn
import time

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5 \nenter layer to run: ")

print("option: ", option)
torch.set_num_threads(1)

# conv12
layerCONV = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(41,11), stride=(2,2), padding=(20,5)),
    nn.BatchNorm2d(32),
    nn.Hardtanh(0, 20, inplace=True),
    nn.Conv2d(32, 32, kernel_size=(21,11), stride=(2,1), padding=(10,5)),
    nn.BatchNorm2d(32),
    nn.Hardtanh(0, 20, inplace=True)
)

# bi-lstm1
layerLSTM1 = nn.Sequential(
    nn.LSTM(1280, 512, bidirectional=True, bias=True)
)

# bi-lstm23456
layerBN1 = nn.BatchNorm1d(1024)
layerLSTM2 = nn.LSTM(1024, 512, bidirectional=True, bias=True)

# fc1
layerBN2 = nn.BatchNorm1d(1024)
layerFC = nn.Linear(1024, 29)

layerCONV = torch.load('./weight/CONV')
layerLSTM1 = torch.load('./weight/LSTM1')
layerLSTM2 = torch.load('./weight/LSTM2')
layerBN1 = torch.load('./weight/BN1')
layerBN2 = torch.load('./weight/BN2')
layerFC = torch.load('./weight/FC')

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
        if i >= 10:
            print(end-start)
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
        if i >= 10:
            print(end-start)
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
        if i >= 10:
            print(end-start)
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
        if i >= 10:
            print(end-start)
            avg_time = avg_time + end - start
    avg_time = avg_time / 100
    print("avg_time: ", avg_time)

elif (option == '5'):
    print("5!")

    avg_time = 0
    for i in range(110):
        x = torch.randn((1, 1, 160, 1151))
        start = time.time() #####
        start_conv = time.time() #####
        x = layerCONV(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)
        end_conv = time.time() #####

        start_lstm1 = time.time() #####
        x, _ = layerLSTM1(x)
        end_lstm1 = time.time() #####

        start_lstm2 = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        end_lstm2 = time.time() #####

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

        start_fc = time.time() #####
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN2(x)
        x = x.view(sizes[0], sizes[1], -1)
        x = layerFC(x)
        end_fc = time.time()   #####
        end = time.time()   #####

        time = end - start
        time_conv = end_conv - start_conv
        time_lstm1 = end_lstm1 - start_lstm1
        time_lstm2 = end_lstm2 - start_lstm2
        time_fc = end_fc - start_fc
        if i >= 10:
            print("conv\t", time_conv)
            print("lstm1\t", time_lstm1))
            print("lstm2\t", time_lstm2, "x5")
            print("fc\t", time_fc)
            print(time)
            avg_conv = avg_conv + end - start
            avg_lstm1 = avg_lstm1 + end - start
            avg_lstm2 = avg_lstm2 + end - start
            avg_fc = avg_fc + end - start
            avg_tot = avg_tot + end - start
    avg_tot = avg_tot / itr
    print("avg_time: ", avg_time)

