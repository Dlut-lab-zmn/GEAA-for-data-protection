#from keras.utils import to_categorical
'''Train CIFAR10 with PyTorch.'''

import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataload_class_net import load_file_list,load_test_list,get_batch,get_test


import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume_path', type=str,default = '', help='resume from checkpoint')#./checkpoint/class_net.pth
parser.add_argument('--load_path', type=str,default = '', help='dataset path')#./checkpoint/class_net.pth
parser.add_argument('--save_name', type=str,default = 'cifar', help='dataset path')#./checkpoint/class_net.pth
parser.add_argument('--dataset', default='cifar', type=str, help='learning rate')
parser.add_argument('--device', default='2', type=str, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nclass', default=10, type=int, 
                    help='number of categories')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.dataset in ['cifar','imagenet','svhn']:
    in_channel = 3
elif args.dataset in ['mnist','fashion']:
    in_channel =1
else:
    print("dataset is not avalible")
    assert args.dataset in ['cifar','mnist','fashion','imagenet','svhn']
# Model
print('==> Building model..')
#net = VGG('VGG16',in_channel,args.nclass)
#
net = ResNet18(in_channel,args.nclass)
# net = PreActResNet18()
# net = GoogLeNet(in_channel)
#net = DenseNet121(in_channel,args.nclass) 
# net = ResNeXt29_2x64d()
#net = MobileNet(in_channel,args.nclass)
#net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume_path:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume_path, map_location='cpu')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'];print(start_epoch)
    net.load_state_dict(checkpoint['net'])


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def cross_entropy_loss(output,label):
    # convert out to softmax probability

    prob = torch.clamp(torch.softmax(output, 1), 1e-10, 1.0)

    loss = torch.mean(-label * torch.log(prob + 1e-8))

    return loss

# Training
def train(train_num, net, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_size = 128
    iters = train_num//batch_size
    for iter in range(iters):
        inputs, targets = get_batch(batch_size)
        inputs = torch.FloatTensor(inputs)
        targets = torch.FloatTensor(targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs,targets)
    
     
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, labels = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(labels).sum().item()
    
        progress_bar(iter, iters, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(iter+1), 100.*correct/total, correct, total))
    #print(torch.sum(sum))
def test(test_num, net, criterion, optimizer, epoch):
    global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        batch_size = 100
        iters = test_num//batch_size
        for iter in range(iters):
            inputs, targets = get_test(batch_size)
            inputs = torch.FloatTensor(inputs)
            targets = torch.FloatTensor(targets)
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = net(inputs)
            
            loss = criterion(outputs,targets)

        
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, labels = targets.max(1)
            total += targets.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar(iter, iters, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(iter+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.save_name + '.pth')
        best_acc = acc
if not args.load_path:
    load_path = './data/' + args.dataset + '_data/' +args.dataset+'_data'
else:
    load_path = args.load_path
train_num = load_file_list(load_path,args.nclass)
test_num = load_test_list(load_path,args.nclass)
criterion = cross_entropy_loss  
test(test_num, net, criterion, optimizer, 0)

for epoch in range(args.epochs):
    train(train_num, net, criterion, optimizer, epoch)
    test(test_num, net, criterion, optimizer, epoch)


optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
for epoch in range(args.epochs):
    train(train_num, net, criterion, optimizer, epoch)
    test(test_num, net, criterion, optimizer, epoch)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
for epoch in range(args.epochs):
    train(train_num, net, criterion, optimizer, epoch)
    test(test_num, net, criterion, optimizer, epoch)

