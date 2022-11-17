import torch
import torch.nn as nn
import torch.nn.functional as F
import pim_library as pim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(32, 4096, bias=False, dtype=torch.bfloat16)
        )

    def forward(self, x):
        x = self.f(x)
        return x

pim_model = MyModel()
cpu_model = MyModel()
cpu_model.load_state_dict(pim_model.state_dict())

x = torch.randn(32, dtype=torch.bfloat16)

pim.init()
pim.to_pim(pim_model)

cpu_out = cpu_model(x)
pim_out = pim_model(x)

print(cpu_model)
print(pim_model)
print(cpu_out)
print(pim_out)
