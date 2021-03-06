# Author Lt Zhao
'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'vgg'
]


class LBP(nn.Module):
    '''
    VGG model
    '''

    def __init__(self):
        super(LBP, self).__init__()
        kernel1 = torch.FloatTensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.FloatTensor([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.FloatTensor([[0., 0., 1.], [0., -1., 0.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.FloatTensor([[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.FloatTensor([[0., 0., 0.], [0., -1., 0.], [0., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.FloatTensor([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel7 = torch.FloatTensor([[0., 0., 0.], [0., -1., 0.], [1., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        kernel8 = torch.FloatTensor([[0., 0., 0.], [1., -1., 0.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        self.kernel1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.kernel2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.kernel3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.kernel4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.kernel5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.kernel6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.kernel7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.kernel8 = nn.Parameter(data=kernel8, requires_grad=False)
        print("----------IN LBP_AUTO_LEARN----------")
        self.init_feature = init_feature()
        self.conv_noise1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv_noise1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        # self.conv_noise2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv_noise3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_noise4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_noise5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_noise6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_noise7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_noise8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.squeeze1 = nn.Conv2d(512, 256, kernel_size=1)
        self.squeeze2 = nn.Conv2d(256, 64, kernel_size=1)
        self.resfeas = res_layers()

        self.res_block1 = res_block([128, 128], 128, batch_norm=False)
        self.squeeze3 = nn.Conv2d(256, 128, kernel_size=1)

        self.res_block2 = res_block([128, 128], 256, batch_norm=False)
        self.res_block3 = res_block([128, 64, 32,8], 256, batch_norm=False)
        self.res_block4 = res_block([128, 128], 256, batch_norm=False)

        self.squeeze4 = nn.Conv2d(256, 128, kernel_size=1)
        self.res_block5 = res_block([64, 32, 8], 128, batch_norm=False)
        self.res_block6 = res_block([128,64, 32,8], 256, batch_norm=False)

        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(18432, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 24),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(18432, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 24),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(18432, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 24),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def get_kernel(self, index):
        if index == 0:
            kernel = self.kernel1
        elif index == 1:
            kernel = self.kernel2
        elif index == 2:
            kernel = self.kernel3
        elif index == 3:
            kernel = self.kernel4
        elif index == 4:
            kernel = self.kernel5
        elif index == 5:
            kernel = self.kernel6
        elif index == 6:
            kernel = self.kernel7
        else:
            kernel = self.kernel8
        return kernel

    def forward(self, x):
        
        x_feature = []
        b,c,h,w = x.shape
        for i in range(c):
            for j in range(8):
                self.kernel = self.get_kernel(j)
                x_feature.append(F.conv2d(x[:, i].unsqueeze(1), weight=self.kernel, padding=1))
        x_features = torch.cat(x_feature,1)
        init_feature = self.init_feature(x)
        content1 = self.conv_noise1(init_feature)
        # content1 = self.conv_noise1(init_feature)
        # content2 = self.conv_noise2(content1)
        # content3 = self.conv_noise3(content2)
        content4 = self.conv_noise4(content1)
        content5 = self.conv_noise5(content4)
        content6 = self.conv_noise6(content5)
        content7 = self.conv_noise6(content6)
        content8 = self.conv_noise6(content7)
        content = torch.cat([content1, content4, content5,
                             content6, content7, content8], 1)
        squee_con1 = self.squeeze1(content)
        squee_con2 = self.squeeze2(squee_con1)
        noise1 = init_feature - squee_con2
        noise2 = self.resfeas(init_feature)
        noise = torch.cat([noise1, noise2], 1)
        fea1 = self.res_block1(noise)
        noise_fea1 = torch.cat([noise, fea1], 1)
        fea2 = self.res_block2(noise_fea1)
        squeeze_noise_fea1 = self.squeeze3(noise_fea1)
        noise_fea2 = torch.cat([squeeze_noise_fea1, fea2], 1)

        conD = self.res_block3(noise_fea2)
        conD = conD.view(conD.size(0), -1)
        conD_Fea = self.classifier1(conD)

        conM = self.res_block4(noise_fea2)
        conM_Fea = self.res_block5(conM)
        conM_Fea = conM_Fea.view(conM_Fea.size(0), -1)
        conM_Fea = self.classifier2(conM_Fea)

        noise_fea2 = self.squeeze4(noise_fea2)
        noise_fea3 = torch.cat([noise_fea2,conM],1)
        conB = self.res_block6(noise_fea3)
        conB_Fea = conB.view(conB.size(0), -1)
        conB_Fea = self.classifier3(conB_Fea)
        return conD_Fea,conM_Fea,conB_Fea


def res_layers(batch_norm=True):
    res = [64, 64]
    layers = []
    in_channels = 64
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


def init_feature(batch_norm=True):
    res = [64, 64]
    layers = []
    in_channels = 24
    for v in res:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 64

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def res_block(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


def vgg():
    """VGG 11-layer model (configuration "A")"""
    return LBP()

