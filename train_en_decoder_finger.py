
import argparse
import time
from dataload_en_decoder import load_file_list, load_file_list2, load_test_list, load_test_list2, \
    get_batch, get_test, get_batch2, get_test2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import en_decoder_finger
import en_decoder
import os
import numpy as np
import random
import cv2
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--repeat_epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to repeat')
parser.add_argument('--bound', default=10, type=int, metavar='N',
                    help='low bound constraint')
parser.add_argument('--gama', default= 1.5 , type=float, metavar='N',
                    help='hyperparameters to control the boundary')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')

parser.add_argument('--arch', dest='arch',
                    help='The flag to choose the model',
                    default='en_decoder_finger', type=str)
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--resume', dest='resume',
                    help='The path to resume the trained model',
                    default='', type=str)
parser.add_argument('--dataset', dest='dataset',
                    help='The path to resume the trained model',
                    default='cifar', type=str)
parser.add_argument('--class_net_path', dest='class_net_path',
                    help='The directory used to save the trained classification models',
                    default='checkpoint', type=str)
parser.add_argument('--load_path', dest='load_path',
                    help='The directory used to store dataset',
                    default='', type=str)
parser.add_argument('--device', help='GPUS',
                    default='0', type=str)
parser.add_argument('--nclass', default=10, type=int, 
                    help='number of categories')

def one_hot(a, n):
    a = a.cpu()
    b = a.shape[0]
    c = np.zeros([b, n])
    for i in range(b):
        c[i][int(a[i])] = 1
    return c


def cross_entropy_loss(output, label):
    # convert out to softmax probability

    prob = torch.clamp(torch.softmax(output, 1), 1e-10, 1.0)

    loss = torch.sum(-label * torch.log(prob + 1e-8))

    return loss


def dataload_train(path):
    train_num = load_file_list(path,args.nclass)
    train_num2 = load_file_list2(path,args.nclass)
    return train_num,train_num2
def dataload_test(path):
    test_num = load_test_list(path,args.nclass)
    test_num2 = load_test_list2(path,args.nclass)
    return test_num,test_num2

def to_att(input,att):
    assert att in ['float','cuda','tensor']
    if isinstance(input, tuple):
        input1, input2,input3 = input
        if att == 'float':
            input1 = input1.float()
            input2 = input2.float()
            input3 = input3.float() 
        elif att == 'cuda':
            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()
        else:        
            input1 = torch.FloatTensor(input1)
            input2 = torch.FloatTensor(input2)
            input3 = torch.FloatTensor(input3)
        return (input1,input2,input3)
    else:
        if att == 'float':
            input = input.float()
        elif att == 'cuda':
            input = input.cuda()
        else:
            input = torch.FloatTensor(input)
        return input

def to_atts(list1,att):
    list2 = []
    for i in range(len(list1)):
        list2.append(to_att(list1[i],att))
    return list2 

def process_data(data,flag):
    if flag in ['cifar','imagenet','svhn']:
        mean = torch.as_tensor(( 0.4914, 0.4822, 0.4465), dtype=torch.float32, device=data.device)
        std = torch.as_tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32, device=data.device)
        data = data.div(255)
        data.sub_(mean[None,:, None, None]).div_(std[None,:, None, None])

    elif flag in ['mnist','fashion']:
        mean = torch.as_tensor([ 0.5], dtype=torch.float32, device=data.device)
        std = torch.as_tensor([0.5], dtype=torch.float32, device=data.device)        
        data = data.div(255)
        data.sub_(mean[None,:, None, None]).div_(std[None,:, None, None])
    return data
def process_datas(datas,flag):
        data1,data2 = datas
        data1 = process_data(data1,flag)
        data2 = process_data(data2,flag)
        return (data1,data2)
def main():
    global args, best_prec1
    args = parser.parse_args()

    assert args.arch in ['en_decoder','en_decoder_finger']

    # choose the device
    os.environ['CUDA_VISIBLE_DEVICES']=args.device

    if args.load_path:
        if os.path.isfile(args.load_path):
            load_path = args.load_path
    else:
        load_path = './data/' + args.dataset + '_data/' +args.dataset+'_data'

    # load datasets
    train_num,train_num2 = dataload_train(load_path)
    test_num,test_num2 = dataload_test(load_path)
    
    # choose model
    if args.arch == 'en_decoder_finger':
        in_model = en_decoder_finger
    else:
        in_model = en_decoder
    print("Now is training the {}".format(args.arch))
    
    # load pre-trained classification model and find optional trained model
    if not args.save_dir:
        args.save_dir = args.arch
    model_init = os.path.join(args.class_net_path, args.dataset + ".pth")
    model_resume = args.resume

    # load init model
    if args.dataset in ['cifar','imagenet','svhn']:
        args.in_channel = 3
        args.out_channel = 1
    elif args.dataset in ['mnist','fashion']:
        args.in_channel = 1
        args.out_channel = 1
    model = in_model.__dict__[args.arch](model_init,args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if model_resume:
        if os.path.isfile(model_resume):
            print("=> loading checkpoint '{}'".format(model_resume))
            checkpoint = torch.load(model_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = cross_entropy_loss  # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # test before train
    if args.evaluate:
        test(test_num, test_num2, model)
        return
    
    # training process
    best_prec1 = 0
    rest_time = 24*3600
    args.balance_value = 0
    for epoch_iter in range(args.repeat_epochs):
      args.gama = args.gama + 0.1
      
      for epoch in range(args.epochs):
        adjust_learning_rate(optimizer,epoch)

        # train for one epoch
        start_time = time.time()
        train(train_num, train_num2, model, criterion, optimizer,epoch,rest_time,flag = args.arch)
        time_len = time.time()-start_time
        rest_time = time_len*( (args.repeat_epochs-epoch_iter - 1)*args.epochs + args.epochs - epoch )
        # since all data is ok for designers, it is no need to set the validate process
        # evaluate on validation set
        #prec1 = validate(test_num, test_num2, model, criterion,epoch,flag = args.arch)
        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        is_best = True
        best_prec1 = 100#max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, str(epoch_iter)+'_'+ args.dataset+'.tar'))

def init_AverageMeter(num):
    top = []
    for i in range(num):
        top.append(AverageMeter())
    return top


def generate_masks(shape):
      """
      mask1s = []
      #index = random.randint(0,6)
      for i in range(shape[0]):
        mask1 = torch.reshape(torch.tensor(np.zeros((shape[2],shape[3]))),(1,1,shape[2],shape[3])).float()
        index = random.randint(0,6)
        tup = [[0.4,0.6,0.25,0.75],[0.2,0.4,0.25,0.75],[0.6,0.8,0.25,0.75],[0.25,0.75,0.6,0.8],[0.25,0.75,0.4,0.6],[0.25,0.75,0.2,0.4],[0.25,0.75,0.25,0.75]]
        mask1[:,:,int(tup[index][0]*shape[2]):int(tup[index][1]*shape[2]),int(tup[index][2]*shape[3]):int(tup[index][3]*shape[3])] = 1.
        mask1s.append(mask1)
      mask1 = torch.cat(mask1s,0)
      """ 
      mask1s = []
      #index = 9#random.randint(1,9)
      for i in range(shape[0]):
        index = random.randint(1,9)
        mask1 = cv2.imread('./tools/'+str(shape[2])+'/'+str(index)+'.png')/255.
        mask1 = torch.reshape(torch.tensor(np.squeeze(np.array(mask1)[:,:,0])),(1,1,shape[2],shape[3])).float()
        mask1s.append(mask1)
      mask1 = torch.cat(mask1s,0)

      return mask1


def train(train_num, train_num2, model, criterion, optimizer, epoch,rest_time,flag = 'auto_learn'):
    """
        Run one train epoch
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = init_AverageMeter(10)

    # switch to train mode
    model.train()
    
    # choose the max num from pair
    iters1 = train_num // args.batch_size
    iters2 = train_num2 // args.batch_size
    if iters1 < iters2:
        iters = iters1
    else:
        iters = iters2
    acc = 0
    for iter in range(iters):
        
        # load batch data
        input1,label1 = get_batch(args.batch_size)
        b,c,h,w = np.array(input1).shape
        #mask1 = torch.reshape(torch.tensor(np.zeros((h,w))),(1,1,h,w)).float()
        #mask1[:,:,int(0.25*h):int(0.75*h),int(0.25*w):int(0.75*w)] = 1.
        #masks1 = mask1.repeat(args.batch_size,1,1,1)
        masks1 = generate_masks((b,c,h,w))
        #mask1 = torch.reshape(torch.tensor(np.array(Image.open('./1.png'))),(1,1,h,w)).float()/255.
        input2,label2 = get_batch2(args.batch_size)#labelp, mlabelp, clabelp
        mask2 = torch.reshape(torch.tensor(np.zeros((h,w))),(1,1,h,w)).float()
        masks2 = mask2.repeat(args.batch_size,1,1,1)
        # process data
        input1,input2,label1,label2 = to_atts([input1,input2,label1,label2],'tensor')
        if args.cpu == False:
            input1,input2,label1,label2,masks1,masks2 = to_atts([input1,input2,label1,label2,masks1,masks2],'cuda')


        # forward
        if args.arch == 'en_decoder_finger':
            adv_g,adv_p,noise_g, noise_p = model(input1,input2,masks1,masks2)
        else:
            adv_g,adv_p,noise_g, noise_p = model(input1,input2)
        pr_adv_g = process_datas(adv_g,args.dataset)
        pr_adv_p = process_datas(adv_p,args.dataset)
        outputg1 = model.net(pr_adv_g[0])
        outputg2 = model.net(pr_adv_g[1])
        outputp1 = model.net(pr_adv_p[0])
        outputp2 = model.net(pr_adv_p[1])
        
        # loss function 
        
        # balance the encoder noise and decoder noise
        noise = torch.mean(abs(noise_g[0] - noise_p[0]))+torch.mean(abs(noise_g[1] - noise_p[1]))
        
        # the size of noise
        noise_g1 = torch.mean(abs(noise_g[0]))
        noise_g2 = torch.mean(abs(noise_g[1]))
        noise_p1 = torch.mean(abs(noise_p[0]))
        noise_p2 = torch.mean(abs(noise_p[1]))
        
        # disrupt the distribution
        loss_g1 = criterion(outputg1, label2)
        loss_g2 = criterion(outputg2, label1)
        loss_p1 = criterion(outputp1, label1) 
        loss_p2 = criterion(outputp2, label2)
        loss_g = loss_g1+loss_g2#max(min(loss_g1,loss_g2),args.batch_size*1.8)
        loss_p = 0
        
        # reconstruct the image from the decoder
        loss_r1 = torch.mean(abs(adv_p[0]-128.-input1))
        loss_r2 = torch.mean(abs(adv_p[1]-128.-input2))

        # finger loss
        if args.arch == 'en_decoder_finger':
            G_mask1 = model.finger_net(adv_p[0]/255.)
            G_mask2 = model.finger_net(adv_p[1]/255.)
            G_mask3 = model.finger_net((input1+128.)/255.)
            G_mask4 = model.finger_net((input2+128.)/255.)
            insert_loss = torch.mean(abs(G_mask1- masks1)+abs(G_mask2- masks1)+abs(G_mask3- masks2)+abs(G_mask4- masks2))
            #insert_loss = torch.mean((G_mask1+G_mask2)*(1-2*masks1) + (G_mask3+G_mask4)*(1-2*masks2) )
            # total loss function 
            loss = noise+ max((loss_g + loss_p)/2,args.gama*args.batch_size)-min(args.bound,noise_p1)-min(args.bound,noise_p2) +(101-args.balance_value)*insert_loss+ max(abs(noise_p2- noise_p1),5)
            #loss = noise+ max((loss_g + loss_p)/2,args.gama*args.batch_size)-min(10,noise_p1)-min(10,noise_p2) +insert_loss
            #loss =  noise+ (max(loss_g1,args.gama*args.batch_size)+max(loss_g2,args.gama*args.batch_size))/2-min(10,noise_p1)-min(10,noise_p2) +insert_loss
        else:
            loss =  noise+ max((loss_g + loss_p)/2,args.gama*args.batch_size)-min(10,noise_p1)-min(10,noise_p2) 
            insert_loss = 0
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # process
        outputg1,outputg2,outputp1,outputp2 = to_atts([outputg1,outputg2,outputp1,outputp2],'float')
        loss = loss.float()
        if args.arch == 'en_decoder_finger':
            G_mask1_copy = G_mask1.cpu().detach()
            masks1_copy = masks1.cpu().detach()
            G_mask3_copy = G_mask3.cpu().detach()
            masks2_copy = masks2.cpu().detach()
            acc += (np.sum(  abs(np.sum(abs(np.array(G_mask1_copy)),(1,2,3))-np.sum(abs(np.array(masks1_copy)),(1,2,3)) )<5) + 
                np.sum( abs( np.sum(abs(np.array(G_mask3_copy)),(1,2,3))-np.sum(abs(np.array(masks2_copy)),(1,2,3))) <5))/2
        # measure accuracy and record loss
        if iter % args.print_freq == 0:
            if args.arch == 'en_decoder_finger':
                print(100*acc/((iter+1)*args.batch_size))
                args.balance_value = 100*acc/((iter+1)*args.batch_size)
            prec0 = accuracy(outputg1.data, label2, 1) # big
            top[0].update(prec0.item(), input1.size(0))
            prec1 = accuracy(outputg2.data, label1, 1) # big
            top[1].update(prec1.item(), input1.size(0))
            
            prec2 = accuracy(outputp1.data, label1, 1) # big
            top[2].update(prec2.item(), input1.size(0))
            prec3 = accuracy(outputp2.data, label2, 1) # big
            top[3].update(prec3.item(), input1.size(0))
            
            
            losses.update(loss.item(), input1.size(0))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@0 {top0:.3f}\t'
                  'Prec@1 {top1:.3f}\t'
                  'Prec@2 {top2:.3f}\t'
                  'Prec@3 {top3:.3f}\n'
                  'input noise {loss1:.3f}\t'
                  'noise_g1 {loss2:.3f}\t'
                  'noise_g2 {loss3:.3f}\t'
                  'max_g {loss4:.3f}\t'
                  'insert_loss {loss5:.3f}\t'
                  'rest_time {rest:.1f} h\t'.format(
                epoch, iter, iters, loss=losses, top0=prec0,top1=prec1, top2=prec2, top3=prec3,
            loss1 = loss_r1 + loss_r2,loss2 = noise_p1,loss3 = noise_p2,loss4 = max(loss_g1,loss_g2),loss5 = insert_loss,rest = rest_time/3600.
            ))


def test(test_num, test_num2, model):

    # switch to evaluate mode
    model.eval()
    end = time.time()
    iters2 = test_num // args.batch_size
    iters3 = test_num2 // args.batch_size
    if iters2 < iters3:
        iters = iters2
    else:
        iters = iters3
    for iter in range(iters):

        # load batch data
        input1,label1 = get_test(args.batch_size)
        input2,label2 = get_test2(args.batch_size)#labelp, mlabelp, clabelp
        
        # process data
        input1,input2,label1,label2 = to_atts([input1,input2,label1,label2],'tensor')
        if args.cpu == False:
            input1,input2,label1,label2 = to_atts([input1,input2,label1,label2],'cuda')


        # compute output
        with torch.no_grad():
            pr_adv_g = process_data(input1+128.,args.dataset)
            pr_adv_p = process_data(input2+128.,args.dataset)
            outputg1 = model.net(pr_adv_g)
            outputp1 = model.net(pr_adv_p)


        outputg1,outputp1 = to_atts([outputg1,outputp1],'float')
        # measure accuracy and record loss

        if iter % args.print_freq == 0:
            # Accuracy t->p  top1~top3
            # Accuracy p->t  top4~top6

            prec0 = accuracy(outputg1.data, label1, 1)
            prec1 = accuracy(outputp1.data, label2, 1)
 
            print('Epoch: [{0}/{1}]\t'
                  'Prec@0 {top0:.3f}\t'
                  'Prec@1 {top1:.3f}\t'.format(
                iter, iters, top0=prec0,top1=prec1))





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // (args.epochs/3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(data1, data2, value):
    temp1 = MaxNum(data1, value)
    temp2 = MaxNum(data2, value)
    return np.mean(acc(temp1, temp2))


def MaxNum(nums, value):
    temp1 = []
    nums = list(nums)
    for i in range(args.batch_size):
        temp = []
        Inf = 0
        nt = list(nums[i])
        for t in range(value):
            temp.append(nt.index(max(nt)))
            nt[nt.index(max(nt))] = Inf
        temp.sort()
        temp1.append(temp)
    return temp1


def acc(temp, index):
    accuracy = []  # print(np.array(temp).shape)
    for k in range(len(temp)):
        accuracy.append((temp[k] == index[k]))
    return accuracy


if __name__ == '__main__':
    main()
