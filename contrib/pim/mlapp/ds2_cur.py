from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

#torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5, all_in_one:6 \nenter layer to run: ")

print("option: ", option)

torch.set_num_threads(8)

max_iter = 20
warm_iter = 2
num_iter = max_iter - warm_iter

# conv12
if (option == '1'):
    conv1 = torch.load('./weight/conv1')
    conv2 = torch.load('./weight/conv2')
    bn1 = torch.load('./weight/bn1')
    bn2 = torch.load('./weight/bn2')
    layerLSTM1 = torch.load('./weight/LSTM1')

# bi-lstm1
elif (option == '2'):
    conv1 = torch.load('./weight/conv1')
    conv2 = torch.load('./weight/conv2')
    bn1 = torch.load('./weight/bn1')
    bn2 = torch.load('./weight/bn2')
    layerLSTM1 = torch.load('./weight/LSTM1')

# bi-lstm23456
elif (option == '3'):
    layerLSTM1 = torch.load('./weight/LSTM1')
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')

# fc1
elif (option == '4'):
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')

# full model
elif (option == '5' or option == '6'):
    conv1 = torch.load('./weight/conv1')
    conv2 = torch.load('./weight/conv2')
    bn1 = torch.load('./weight/bn1')
    bn2 = torch.load('./weight/bn2')
    layerLSTM1 = torch.load('./weight/LSTM1')
    layerBN1 = torch.load('./weight/BN1')
    layerLSTM2 = torch.load('./weight/LSTM2')
    layerBN2 = torch.load('./weight/BN2')
    layerFC = torch.load('./weight/FC')
hardtanh = nn.Hardtanh(0, 20, inplace=True)

print("\n----lets run!----")

def run_conv():
    print("compute: convolution layer 1 and 2")

    avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))

        start = time.time() #####
        x = hardtanh(bn2(conv2(hardtanh(bn1(conv1(x))))))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)
        end = time.time()   #####

        x, _ = layerLSTM1(x)

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
        x = torch.randn((1, 32, 80, 576))
        x = hardtanh(bn2(conv2(x)))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)

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
    bn_avg_time = 0
    lstm_avg_time = 0
    print("iter\t time")
    print("total, bn, lstm2")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1280)) ##
        x, _ = layerLSTM1(x)

        start = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        bn_start = time.time() # bn>>
        x = layerBN1(x)
        bn_end = time.time() # <<bn
        x = x.view(sizes[0], sizes[1], -1)
        lstm_start = time.time() # lstm>>
        x, _ = layerLSTM2(x)
        lstm_end = time.time() # <<lstm
        end = time.time()   #####

        print(i, "\t", end-start, "     \t", bn_end-bn_start, "     \t", lstm_end - lstm_start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
            bn_avg_time = bn_avg_time + bn_end - bn_start
            lstm_avg_time = lstm_avg_time + lstm_end - lstm_start
            
    avg_time = avg_time / num_iter
    bn_avg_time = bn_avg_time / num_iter
    lstm_avg_time = lstm_avg_time / num_iter

    print("avg_time: ", avg_time)
    print("bn_avg_time: ", bn_avg_time)
    print("lstm_avg_time: ", lstm_avg_time)
    return avg_time

def run_fc():
    print("compute: fc layer")
    
    avg_time = 0
    bn_avg_time = 0
    fc_avg_time = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1024))
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)

        start = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        bn_start = time.time() # bn>>
        x = layerBN2(x)
        bn_end = time.time() # <<bn
        x = x.view(sizes[0], sizes[1], -1)
        fc_start = time.time() # fc>>
        x = layerFC(x)
        fc_end = time.time() # <<fc
        end = time.time()   #####
        print(i, "\t", end-start, "     \t", bn_end-bn_start, "     \t", fc_end - fc_start)
        if i >= warm_iter:
            avg_time = avg_time + end - start
            bn_avg_time = bn_avg_time + bn_end - bn_start
            fc_avg_time = fc_avg_time + fc_end - fc_start
    avg_time = avg_time / num_iter
    bn_avg_time = bn_avg_time / num_iter
    fc_avg_time = fc_avg_time / num_iter
    print("avg_time: ", avg_time)
    print("bn_avg_time: ", bn_avg_time)
    print("fc_avg_time: ", fc_avg_time)
    return avg_time

def run_full_model():
    print("conpute: full ds2 model")
    
    a = time.time()
    avg_tot = 0
    avg_conv = 0
    avg_lstm1 = 0
    avg_lstm2 = 0
    avg_fc = 0
    print("iter\t time")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))
        start = time.time() #####
        start_conv = time.time() #####
        x = hardtanh(bn2(conv2(hardtanh(bn1(conv1(x))))))
        end_conv = time.time()   #####
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)

        start_lstm1 = time.time() #####
        x, _ = layerLSTM1(x)
        end_lstm1 = time.time()   #####

        start_lstm2 = time.time() #####
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)
        end_lstm2 = time.time()   #####

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

        time_tot = end - start
        time_conv = end_conv - start_conv
        time_lstm1 = end_lstm1 - start_lstm1
        time_lstm2 = end_lstm2 - start_lstm2
        time_fc = end_fc - start_fc
        if i >= warm_iter:
            print("conv\t", time_conv)
            print("lstm1\t", time_lstm1)
            print("lstm2\t", time_lstm2, "x5")
            print("fc\t", time_fc)
            print("total\t", time_tot)
            avg_conv = avg_conv + time_conv
            avg_lstm1 = avg_lstm1 + time_lstm1
            avg_lstm2 = avg_lstm2 + time_lstm2
            avg_fc = avg_fc + time_fc
            avg_tot = avg_tot + time_tot
    avg_conv = avg_conv / num_iter
    avg_lstm1 = avg_lstm1 / num_iter
    avg_lstm2 = avg_lstm2 / num_iter
    avg_fc = avg_fc / num_iter
    avg_tot = avg_tot / num_iter

    print("\niter : ", i)
    print("avg_conv: ", avg_conv)
    print("avg_lstm1: ", avg_lstm1)
    print("avg_lstm2: ", avg_lstm2, "x5")
    print("avg_fc: ", avg_fc)
    print("avg_tot: ", avg_tot)
    return avg_tot

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
