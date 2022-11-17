from unicodedata import bidirectional
from hwcounter import Timer, count, count_end
import torch
import torch.nn as nn
import os

#torch.set_default_dtype(torch.bfloat16)
torch.set_grad_enabled(False)

option = input("conv12:1, bi-lstm1:2, bi-lstm23456:3, fc1:4, full_model:5, all_in_one:6 \nenter layer to run: ")

print("option: ", option)

max_iter = 40
warm_iter = 5
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

os.system('m5 exit')
os.system('echo CPU Switched!')
torch.set_num_threads(8)
print("\n----lets run!----")

def run_conv():
    print("compute: convolution layer 1 and 2")

    avg_cycle = 0
    print("iter\t cycle")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))

        start = count() #####
        x = hardtanh(bn2(conv2(hardtanh(bn1(conv1(x))))))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)
        end = count_end() #####

        x, _ = layerLSTM1(x)

        print(i, "\t", end - start)
        if i >= warm_iter:
            avg_cycle = avg_cycle + end - start
    avg_cycle = avg_cycle / num_iter
    print("avg_cycle: ", avg_cycle)
    return avg_cycle

def run_lstm1():
    print("compute: lstm layer 1")

    avg_cycle = 0
    print("iter\t cycle")
    for i in range(max_iter):
        x = torch.randn((1, 32, 80, 576))
        x = hardtanh(bn2(conv2(x)))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1)

        start = count() #####
        x, _ = layerLSTM1(x)
        end = count_end() #####

        print(i, "\t", end - start)
        if i >= warm_iter:
            avg_cycle = avg_cycle + end - start
    avg_cycle = avg_cycle / num_iter
    print("avg_cycle: ", avg_cycle)
    return avg_cycle

def run_lstm2():
    print("compute: lstm layer 2 (or 3 4 5 6)")

    avg_cycle = 0
    bn_avg_cycle = 0
    lstm_avg_cycle = 0
    print("iter\t cycle")
    print("Total, bn, lstm2")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1280)) ##
        x, _ = layerLSTM1(x)

        start = count()
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        bn_start = count() # bn>>
        x = layerBN1(x)
        bn_end = count() # <<bn
        x = x.view(sizes[0], sizes[1], -1)
        lstm_start = count() # lstm>>
        x, _ = layerLSTM2(x)
        lstm_end = count() # <<lstm
        end = count_end()

        print(i, "\t", end-start, "     \t", bn_end-bn_start, "     \t", lstm_end - lstm_start)
        if i >= warm_iter:
            avg_cycle = avg_cycle + end - start
            bn_avg_cycle = bn_avg_cycle + bn_end - bn_start
            lstm_avg_cycle = lstm_avg_cycle + lstm_end - lstm_start
    avg_cycle = avg_cycle / num_iter
    bn_avg_cycle = bn_avg_cycle / num_iter
    lstm_avg_cycle = lstm_avg_cycle / num_iter

    print("avg_cycle: ", avg_cycle)
    print("bn_avg_cycle: ", bn_avg_cycle)
    print("lstm_avg_cycle: ", lstm_avg_cycle)
    return avg_cycle

def run_fc():
    print("compute: fc layer")

    avg_cycle = 0
    bn_avg_cycle = 0
    fc_avg_cycle = 0
    print("iter\t cycle")
    print("Total, bn, fc")
    for i in range(max_iter):
        x = torch.randn((576, 1, 1024))
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        x = layerBN1(x)
        x = x.view(sizes[0], sizes[1], -1)
        x, _ = layerLSTM2(x)

        start = count()
        sizes = x.size()
        x = x.view(sizes[0]*sizes[1], -1)
        bn_start = count() # bn>>
        x = layerBN2(x)
        bn_end = count() # <<bn
        x = x.view(sizes[0], sizes[1], -1)
        fc_start = count() # fc>>
        x = layerFC(x)
        fc_end = count() # <<fc
        end = count_end()

        print(i, "\t", end-start, "     \t", bn_end-bn_start, "     \t", fc_end - fc_start)
        if i >= warm_iter:
            avg_cycle = avg_cycle + end - start
            bn_avg_cycle = bn_avg_cycle + bn_end - bn_start
            fc_avg_cycle = fc_avg_cycle + fc_end - fc_start
    avg_cycle = avg_cycle / num_iter
    bn_avg_cycle = bn_avg_cycle / num_iter
    fc_avg_cycle = fc_avg_cycle / num_iter
    avg_cycle = bn_avg_cycle + fc_avg_cycle
    print("avg_cycle: ", avg_cycle)
    print("bn_avg_cycle: ", bn_avg_cycle)
    print("fc_avg_cycle: ", fc_avg_cycle)
    return avg_cycle

def run_full_model():
    print("conpute: full ds2 model")

    avg_cycle = 0
    print("iter\t cycle")
    for i in range(max_iter):
        x = torch.randn((1, 1, 160, 1151))
        start = count() #####
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
        end = count_end()   #####

        print(i, "\t", end-start)
        if i >= warm_iter:
            avg_cycle = avg_cycle + end - start
    avg_cycle = avg_cycle / num_iter
    print("avg_cycle: ", avg_cycle)
    return avg_cycle

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
    conv_cycle = run_conv()
    lstm1_cycle = run_lstm1()
    lstm2_cycle = run_lstm2()
    fc_cycle = run_fc()
    full_model_cycle = run_full_model()

    print("\nTotal Result")
    print("conv layer:\t", conv_cycle)
    print("lstm1 layer:\t", lstm1_cycle)
    print("lstm2 layer:\t", lstm2_cycle)
    print("fc layer:\t", fc_cycle)
    print("full model:\t", full_model_cycle)

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
