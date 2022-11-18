"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import math
#import custom
from . import ChannelPruning, SwitchableBatchNorm2d
#from custom import ChannelPruning, SwitchableBatchNorm2d

__all__ = ['mobilenetv2']

global partition_point
partition_point = False

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, pruning: bool = False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.pruning = pruning
        self.channel_pruning = ChannelPruning(oup, oup)

    def forward(self, x):
        global partition_point
        if self.identity:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        #return out

        if partition_point == True and self.pruning is not False and self.channel_pruning.rate != 0:
            #print("pruning")
            return out * self.channel_pruning(out)
        else:
            #print("not pruning")
            return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.pruning = True
        self.partition = 100
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv0 = conv_3x3_bn(3, input_channel, 2)
        # building inverted residual blocks

        block = InvertedResidual
        i_ = 0
        for t, c, n, s in self.cfgs:
            i_+=1
            layers=[]
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                if i == n-1:
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t,pruning=True))
                else:
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t,pruning=False))
                input_channel = output_channel
            globals()['layer'+str(i_)]=layers
            print(i_)
        #self.features = nn.Sequential(*layers)
        # building last several layers
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*layer6)
        self.layer7 = nn.Sequential(*layer7)

        """
        self.layer1 = self._make_layer(input_channel, block, self.cfgs[0], width_mult,self.pruning)
        self.layer2 = self._make_layer(input_channel, block, self.cfgs[1], width_mult,self.pruning)
        self.layer3 = self._make_layer(input_channel, block, self.cfgs[2], width_mult,self.pruning)
        self.layer4 = self._make_layer(input_channel, block, self.cfgs[3], width_mult,self.pruning)
        self.layer5 = self._make_layer(input_channel, block, self.cfgs[4], width_mult,self.pruning)
        self.layer6 = self._make_layer(input_channel, block, self.cfgs[5], width_mult,self.pruning)
        self.layer7 = self._make_layer(input_channel, block, self.cfgs[6], width_mult,self.pruning)
	"""
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv1 = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()
        self.pruning = 2

    def packing(self, x):
        list = torch.nonzero(t.squeeze()).squeeze()
        package = torch.index_select(x,1,list).view(len(list),-1)
        return torch.cat((package,list.unsqueeze(1)),1)

    def unpacking(self,x,channel,width):
        list = package[:,-1].int()
        matrix = package[:,:-1].view(1,-1,width,width)

        feature = torch.zeros(1,channel,width,width)
        return feature.index_add_(1,list,matrix)

    def forward(self, x) -> torch.Tensor:
        global partition_point
        with torch.autograd.set_detect_anomaly(False):
        #x = self.features(x)
            x = self.conv0(x)

            if self.partition == 1.0:
                partition_point=True
            x = self.layer1(x)

            if self.partition == 1.0:
                if partition_point==True:
                    #x = packing(x)
                    partition_point=False
                    #####HW 변경선####
                    #x = unpacking(x,16,112)

            elif self.partition == 2.0:
                partition_point=True

            x = self.layer2(x)

            if self.partition == 2.0:
                partition_point=False
            elif self.partition == 3.0:
                partition_point=True

            x = self.layer3(x)
            if self.partition == 3.0:
                partition_point=False
            elif self.partition == 4.0:
                partition_point=True

            x = self.layer4(x)
            if self.partition == 4.0:
                partition_point=False
            elif self.partition == 5.0:
                partition_point=True

            x = self.layer5(x)
            if self.partition == 5.0:
                partition_point=False
            elif self.partition == 6.0:
                partition_point=True

            x = self.layer6(x)
            x = self.layer7(x)
            x = self.conv1(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

