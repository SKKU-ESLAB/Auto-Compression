import torch.nn as nn
from functions import *

class alexnet(nn.Module):
    def __init__(self, lq=False):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LQ_Conv2d(64, 192, kernel_size=5, padding=2, lq=lq),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LQ_Conv2d(192, 384, kernel_size=3, padding=1, lq=lq),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            LQ_Conv2d(384, 256, kernel_size=3, padding=1, lq=lq),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            LQ_Conv2d(256, 256, kernel_size=3, padding=1, lq=lq),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x