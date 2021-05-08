# Author Lt Zhao
'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch

__all__ = [
    'u_net_mnist'
]




class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

# 32 16 
class U_net_mnist(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(U_net_mnist, self).__init__()

        self.conv1 = DoubleConv(in_ch,32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.linear4 = nn.Linear(7*7*128, 4*4*128)
        self.linear5 = nn.Linear(4*4*128, 7*7*128)
        self.conv5 = DoubleConv(256, 128)
        self.up6 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        self.conv8 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x) # 28 28 1 32
        p1 = self.pool1(c1) # 14 14 32
        c2 = self.conv2(p1) # 14 14 64
        p2 = self.pool2(c2) # 7 7 64
        c3 = self.conv3(p2) # 7 7 128
        v3 = c3.view(c3.size(0), -1) # 7* 7 *128
        l4 = self.linear4(v3)
        c4 = self.linear5(l4)# 7*7*128
        v4 = c4.view(c3.size()) # 7* 7 *128
        merge5 = torch.cat([v4, c3], dim=1) # 7 * 7 * 256
        c5 = self.conv5(merge5) # 7*7* 128
        up_6 = self.up6(c5) # 14*14* 64
        merge6 = torch.cat([up_6, c2], dim=1) # 14*14 128 
        c6 = self.conv6(merge6) # 14*14*64
        up_7 = self.up7(c6) # 28*28* 32 
        merge7 = torch.cat([up_7, c1], dim=1) # 28*28*64
        c7 = self.conv7(merge7) # 28*28*32
        c8 = self.conv8(c7) # 28*28*1
        out = nn.Sigmoid()(c8)

        return out



def u_net_mnist(in_ch, out_ch):
    """VGG 11-layer model (configuration "A")"""
    return U_net_mnist(in_ch, out_ch)


 