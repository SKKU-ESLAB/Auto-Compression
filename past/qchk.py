import torch
from cpp.pim import PIM_Linear, PIM_BatchNorm1d, PIM_LSTM
import pim_cpp

# input example
in0 = torch.randn(4096)
in1 = torch.randn(4096)
in2 = torch.randn(8192)
in3 = torch.randn(8192)

# pim custom layer
pim_bn = PIM_BatchNorm1d(4096)
pim_bn.weight0 = torch.randn(4096)
pim_bn.weight1 = torch.randn(4096)

pim_fc = PIM_Linear(4096,8192)
pim_fc.weight = torch.randn(4096, 8192)

pim_lstm = PIM_LSTM(4096, 8192)
pim_lstm.weight = torch.randn(4096, 8192*4)

# pim computation
add_out = pim_cpp.add_forward(in0, in1, 4096)
mul_out = pim_cpp.mul_forward(in0, in1, 4096)
bn_out = bn(in0)
fc_out = fc(in0)
lstm_out = lstm(in0, in2, in3)

# print result
print(add_out)
print(mul_out)
print(fc_out)
print(bn_out)
print(lstm_out)

"""
X = torch.randn(1, 2)
h = torch.randn(1, 3)
C = torch.randn(1, 3)
rnn = LLTM(2, 3)

new_h, new_C = rnn(X, (h, C))
(new_h.sum() + new_C.sum()).backward()

for _ in range(5):
    rnn.zero_grad()
    new_h, new_C = rnn(X, (h, C))
    (new_h.sum() + new_C.sum()).backward()
"""
print("ha!")
