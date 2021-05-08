#Author Lt Zhao

from models import *
import torch
import torch.nn as nn
__all__ = [
    'auto_learn',
]
class Auto_learn(nn.Module):
    def __init__(self,resume,attribute,joint,joint2,args):
        super(Auto_learn, self).__init__()
        self.net =ResNet18(args.in_channel,args.nclass).cuda()
        self.net = torch.nn.DataParallel(self.net)
        checkpoint = torch.load(resume, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
        
        for k,v in self.net.named_parameters():
                v.requires_grad=False
        
        self.attribute = attribute
        self.joint = joint
        self.joint2 = joint2

    def forward(self, input1,input2):
    
        att_1 = self.attribute(input1)
        att_2 = self.attribute(input2)
        
        fea = torch.cat((att_1,att_2),1)
        att_1 = self.joint(fea)
        att_2 = self.joint2(fea)
        
        self.output1 = torch.clamp(input1 +att_1+128.,0,255)
        self.output2 = torch.clamp(input2 +att_2+128.,0,255)
        
        return self.output1,self.output2,att_1,att_2

def attribute(in_channels,batch_norm=False):
    res = [64, 64, 64]
    layers = []
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def joint(out_channels,batch_norm=False):
    res = [128, 128, int(out_channels)]
    layers = []
    in_channels = 128
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def joint2(out_channels,batch_norm=False):
    res = [128, 128, int(out_channels)]
    layers = []
    in_channels = 128
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def auto_learn(resume,args):
    """VGG 11-layer model (configuration "A")"""
    return Auto_learn(resume,attribute(args.in_channel),joint(args.in_channel),joint2(args.in_channel),args)