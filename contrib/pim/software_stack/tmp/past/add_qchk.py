import torch
from cpp.pim import PIM_Linear, PIM_BatchNorm1d, PIM_LSTM
import pim_cpp

# input example
a = torch.ones(4096, dtype=torch.int32)
b = torch.ones(4096, dtype=torch.int32)

# computation example
c = pim_cpp.add_forward(a, b)

# print result
print(c)

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
