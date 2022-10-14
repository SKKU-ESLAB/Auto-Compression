import torch

Z = ()
U = ()
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

for name, param in model.named_parameters():
    if name.split('.')[-1] == 'weight':
        print(name)
        print(param.shape)
        Z += (param.detach().cpu().clone(),)
        U += (torch.zeros_like(param).cpu(),)

import pdb; pdb.set_trace()
