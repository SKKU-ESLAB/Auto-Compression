import torch
import time
device = torch.device('cpu')

x = torch.rand(1024*1024*4, dtype=torch.half)
y = torch.rand(1024*1024*4, dtype=torch.half)

x.to(device)
y.to(device)

start = time.time()
z = x + y
end = time.time()

print(z)
print(end - start, "sec")
