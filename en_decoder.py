#Author Lt Zhao
import auto_learn
import torch
import torch.nn as nn
__all__ = [
    'en_decoder',
]
class En_decoder(nn.Module):
    def __init__(self,resume,attribute,degrade,degrade2,in_channels,out_channels):
        super(En_decoder, self).__init__()

        self.generate_net = auto_learn.__dict__['auto_learn'](resume,in_channels,out_channels)
        self.net = self.generate_net.net

        self.attribute = attribute
        self.degrade = degrade
        self.degrade2 = degrade2
    def forward(self, input1,input2,finger1 = None,finger2 = None):
        
        adv1,adv2,n1,n2 = self.generate_net(input1,input2)
        
        att_1 = self.attribute(adv1-128.)
        att_2 = self.attribute(adv2-128.)
        fea = torch.cat((att_1,att_2),1)
        
        att_1 = self.degrade(fea)
        att_2 = self.degrade2(fea)
        
        output1 = torch.clamp(adv1 -att_1,0,255)
        output2 = torch.clamp(adv2 -att_2,0,255)
        return (adv1,adv2),(output1,output2),(n1,n2),(att_1,att_2)
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
def degrade(out_channel, batch_norm=False):
    res = [128, 64, int(out_channel)]
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
def degrade2(out_channel,batch_norm=False):
    res = [128, 64, int(out_channel)]
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

def en_decoder(resume,in_channels,out_channels):
    """VGG 11-layer model (configuration "A")"""
    return En_decoder(resume,attribute(in_channels),degrade(in_channels),degrade2(in_channels),in_channels,out_channels)