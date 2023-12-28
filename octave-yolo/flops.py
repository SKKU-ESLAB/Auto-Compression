import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore")

import model.oct_yolov4
import model.oct_yolov4_v2
import model.oct_yolov4_octconv2
import model.yolov4
from torchsummaryX import summary

device = torch.device('cuda')

"""net = model.yolov4.YOLOv4(448, 20).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
#print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print(summary(net, torch.rand(1,3,448,448).to(device)))


net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.125).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.25).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.375).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.5).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.625).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.75).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.oct_yolov4_v2.YOLOv4(448, 20, 0.875).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print(summary(net, torch.rand(1,3,448,448).to(device)))"""


net = model.oct_yolov4_octconv2.YOLOv4(448, 20, 0.5).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#print(summary(net, torch.rand(1,3,448,448).to(device)))

net = model.yolov4.YOLOv4(448, 20).to(device)
macs, params = get_model_complexity_info(net, (3, 448, 448), as_strings=True,
                                         print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))