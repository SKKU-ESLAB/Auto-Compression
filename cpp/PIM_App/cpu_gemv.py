import torch
import time
device = torch.device('cpu')

A = torch.rand(2048, 4096, dtype=torch.half)
x = torch.rand(1, 2048, dtype=torch.half)

A.to(device)
x.to(device)

start = time.time()
y = torch.mm(x,A)
end = time.time()

print(y)
print(y.size())
print(end - start, "sec")
