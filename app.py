import torch
import torch.nn as nn
import torch.nn.functional as F
import pim_library as pim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(10, 20)
        )

    def forward(self, x):
        x = self.f(x)
        return x

model = MyModel()

x = torch.randn(10)

pim.init()
pim.to_pim(model)
print(model)

x = model(x)

print(x)
