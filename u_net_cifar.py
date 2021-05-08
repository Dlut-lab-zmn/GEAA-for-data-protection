# Author Lt Zhao
'''
Modified from https://github.com/pytorch/vision.git
'''

import torch.nn as nn
import torch

__all__ = [
    'u_net_cifar'
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
class U_net_cifar(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(U_net_cifar, self).__init__()


        self.conv1 = DoubleConv(in_ch,32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)

        self.up5 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv5 = DoubleConv(256, 128)
        self.up6 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv7 = DoubleConv(64, 32)
        self.conv8 = nn.Conv2d(32, out_ch, 1)
 
    def forward(self, x):
        #b,c,h,w = x.shape
        #x = x.resize_((b,c,32,32))
        c1 = self.conv1(x) # 32 32 3 32
        p1 = self.pool1(c1) # 16 16 32
        c2 = self.conv2(p1) # 16 16 64
        p2 = self.pool2(c2) # 8 8 64
        c3 = self.conv3(p2) # 8 8 128
        p3 = self.pool3(c3) # 4 4 128
        c4 = self.conv4(p3) # 4 4 256
        up_5 = self.up5(c4) # 8 8 128
        merge5 = torch.cat([up_5, c3], dim=1) # 8 8 256 
        c5 = self.conv5(merge5) # 8 8 128
        up_6 = self.up6(c5) # 16 16 64
        merge6 = torch.cat([up_6, c2], dim=1) # 16 16 128 
        c6 = self.conv6(merge6) # 16 16 64
        up_7 = self.up7(c6) # 32 32 32 
        merge7 = torch.cat([up_7, c1], dim=1) # 32 32 64
        c7 = self.conv7(merge7) # 32 32 32
        c8 = self.conv8(c7) # 32 32 3
        out = nn.Sigmoid()(c8)
        #out = out.resize_((b,c,h,w))

        return out



def u_net_cifar(in_ch, out_ch):
    """VGG 11-layer model (configuration "A")"""
    return U_net_cifar(in_ch, out_ch)

