import torch
from pim import PIM_Linear, PIM_BatchNorm1d, PIM_LSTM
import pim_cpp

# input example
a = torch.ones(4096, dtype=torch.int32)
b = torch.ones(4096, dtype=torch.int32)

# computation example
c = pim_cpp.add_forward(a, b)

# print result
print(c)
