#Author Lt Zhao
import auto_learn
import torch
import torch.nn as nn
import u_net_cifar
import u_net_mnist
__all__ = [
    'en_decoder_finger',
]
class En_decoder_finger(nn.Module):
    def __init__(self,resume,attribute,degrade,degrade2,args):
        super(En_decoder_finger, self).__init__()

        if 'cifar' in resume or 'imagenet' in resume or 'svhn' in resume:
            u_net = u_net_cifar
            arch = 'u_net_cifar'
        elif 'mnist' in resume or 'fashion' in resume:
            u_net = u_net_mnist
            arch = 'u_net_mnist'
        print(arch)
        self.generate_net = auto_learn.__dict__['auto_learn'](resume,args)
        self.net = self.generate_net.net

        self.attribute = attribute
        self.degrade = degrade
        self.degrade2 = degrade2
        
        # fingernet
        self.finger = DoubleConv(args.out_channel,32)
        self.finger_net = u_net.__dict__[arch](args.in_channel,args.out_channel)
        

    def forward(self, input1,input2,finger1 = None,finger2 = None):
        
        adv1,adv2,n1,n2 = self.generate_net(input1,input2)
        
        att_1 = self.attribute(adv1-128.)
        att_2 = self.attribute(adv2-128.)
        att_3 = self.finger(finger1)
        att_4 = self.finger(finger2)
        fea = torch.cat((att_1,att_2,att_3,att_4),1)
        
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
def degrade(out_channels, batch_norm=False):
    res = [128, 64, int(out_channels)]
    layers = []
    in_channels = 192
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
def degrade2(out_channels,batch_norm=False):
    res = [128, 64, int(out_channels)]
    layers = []
    in_channels = 192
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

def en_decoder_finger(resume,args):
    """VGG 11-layer model (configuration "A")"""
    return En_decoder_finger(resume,attribute(args.in_channel),degrade(args.in_channel),degrade2(args.in_channel),args)