import torch
from torchvision import datasets, transforms
#import helper
import cv2
import os
import numpy as np
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('./', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('./', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
path  = '/data/zhaomengnan/data/fashion_data/fashion_data/'
for i,(img,lab) in enumerate(trainloader):
    tr_path  = path + 'train/'
    print(i)
    print(len(lab))
    for j in range(len(lab)):
        #print(os.path.join(tr_path + str(int(lab[j])), str(i)+'.png'))
        cv2.imwrite(os.path.join(tr_path + str(int(lab[j])), str(i)+'_'+str(j)+'.png'), np.squeeze(np.array(255.*(img[j]*0.5+0.5) )))
for i,(img,lab) in enumerate(testloader):
    te_path  = path + 'test/'
    for j in range(len(lab)):
        cv2.imwrite(os.path.join(te_path + str(int(lab[j])), str(i)+'_'+str(j)+'.png'), np.squeeze(np.array(255.*(img[j]*0.5+0.5)) ))
