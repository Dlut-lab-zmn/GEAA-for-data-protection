
import argparse
from dataload_en_decoder import load_file_list, load_file_list2, load_test_list, load_test_list2, \
    get_batch, get_test, get_batch2, get_test2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import en_decoder
import en_decoder_finger
import os
import numpy as np
import cv2
import random
from PIL import Image
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--repeat_epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to repeat')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
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
parser.add_argument('--class_net_path', dest='class_net_path',
                    help='The directory used to save the trained models',
                    default='checkpoint', type=str)
parser.add_argument('-nclass', '--nclass', default=10, type=int, help='nclass (default: 10)')
parser.add_argument('--dataset', dest='dataset',
                    help='The directory used to store dataset',
                    default='cifar_data', type=str)
parser.add_argument('--load_path', dest='load_path',
                    help='The directory used to store dataset',
                    default='', type=str)
parser.add_argument('--device', help='GPUS',
                    default='3', type=str)
parser.add_argument('--resume', dest='resume',
                    help='The path to resume the trained model',
                    default='', type=str)
parser.add_argument('--arch', dest='arch',
                    help='The flag to choose the model',
                    default='en_decoder_finger', type=str)

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


def dataload_train(path):
    train_num = load_file_list(path)
    train_num2 = load_file_list2(path)
    return train_num,train_num2
def dataload_test(path):
    test_num = load_test_list(path)
    test_num2 = load_test_list2(path)
    return test_num,test_num2
def main():
    global args
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.device
    # Check the save_dir exists or not
    if args.load_path:
        if os.path.isfile(args.load_path):
            load_path = args.load_path
    else:
        load_path = './data/' + args.dataset + '_data/' +args.dataset+'_data'
    train_num,train_num2 = dataload_train(load_path)
    test_num,test_num2 = dataload_test(load_path)



    # choose model
    if args.arch == 'en_decoder_finger':
        in_model = en_decoder_finger
    else:
        in_model = en_decoder
    print("Now is training the {}".format(args.arch));



    # load pre-trained classification model and find optional trained model
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

    # train for one epoch
    validate(train_num, train_num2, model,dataset = 'train',flag = args.arch)
    #validate(test_num, test_num2, model, criterion,epoch,flag = args.arch)
    # evaluate on validation set

def save_imgs(save_path,adv):
    adv_img = torch.squeeze(adv)
    if len(adv_img.shape) == 2:
        pass
    else:
        adv_img = adv_img.transpose(0, 1).transpose(1, 2).contiguous()        

    adv_img = np.float32(adv_img.cpu().detach())
    adv_img = Image.fromarray(np.uint8(adv_img))
    adv_img.save(save_path)
    #cv2.imwrite(save_path,adv_img)
    #print(cv2.imread(save_path) - adv_img)
    #print(np.array(Image.open(save_path))-adv_img)

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



def validate(test_num, test_num2, model,dataset = 'test',flag = 'auto_learn'):
    """
    Run evaluation
    """
    assert flag in ['en_decoder','en_decoder_finger']
    adv_save_train_path = './data/' + args.dataset + '_data/adv/train/'
    process_save_train_path = './data/' +args.dataset + '_data/process/train/'
    acc = 0
    total_noise = 0
    total_Rnoise = 0
    # switch to train mode, eval is not available for this task, we need the avarage value of the input instead of the default value
    model.train()
    iters2 = test_num // args.batch_size
    iters3 = test_num2 // args.batch_size
    if iters2 < iters3:
        iters = iters2
    else:
        iters = iters3
    print(iters)
    for iter in range(iters):

        if dataset == 'train':
            input1,label1 = get_batch(args.batch_size)
            input2,label2 = get_batch2(args.batch_size)
        else:
            input1,label1 = get_test(args.batch_size)
            input2,label2 = get_test2(args.batch_size)
        b,c,h,w = np.array(input1).shape
        #mask1 = torch.reshape(torch.tensor(np.zeros((h,w))),(1,1,h,w)).float()
        #mask1[:,:,int(0.25*h):int(0.75*h),int(0.25*w):int(0.75*w)] = 1.
        #masks1 = mask1.repeat(args.batch_size,1,1,1)
        masks1 = generate_masks((b,c,h,w))
        mask2 = torch.reshape(torch.tensor(np.zeros((h,w))),(1,1,h,w)).float()
        masks2 = mask2.repeat(args.batch_size,1,1,1)
        # process data
        input1,input2,label1,label2 = to_atts([input1,input2,label1,label2],'tensor')
        if args.cpu == False:
            input1,input2,label1,label2,masks1,masks2 = to_atts([input1,input2,label1,label2,masks1,masks2],'cuda')


        # compute output
        #with torch.no_grad():
        if args.arch == 'en_decoder_finger':
            adv_g,adv_p,noise_g, noise_p = model(input1,input2,masks1,masks2)
        else:
            adv_g,adv_p,noise_g, noise_p = model(input1,input2)

        if args.arch == 'en_decoder_finger':

            G_mask1 = model.finger_net(adv_p[0]/255.)
            G_mask2 = model.finger_net(adv_p[1]/255.)
            G_mask3 = model.finger_net((input1+128.)/255.)
            G_mask4 = model.finger_net((input2+128.)/255.)

            G_mask1_copy = G_mask1.cpu().detach()
            masks1_copy = masks1.cpu().detach()
            G_mask3_copy = G_mask3.cpu().detach()
            masks2_copy = masks2.cpu().detach()
            acc += (np.sum(  abs(np.sum(abs(np.array(G_mask1_copy)),(1,2,3))-np.sum(abs(np.array(masks1_copy)),(1,2,3)) )<5) + 
                np.sum( abs( np.sum(abs(np.array(G_mask3_copy)),(1,2,3))-np.sum(abs(np.array(masks2_copy)),(1,2,3))) <5))/2

        index1 = torch.argmax(label1,1)
        index2 = torch.argmax(label2,1)
        for i in range(args.batch_size):
            adv_1 = adv_g[0]
            adv_2 = adv_g[1]
            pro_1 = adv_p[0]
            pro_2 = adv_p[1]
            #print(os.path.join(os.path.join(adv_save_train_path,str(args.nclass-1-int(index1[i]))),str(iter)+'_'+str(i)+'.png'))
            #save_imgs(os.path.join('./data/fashion_data/ori',str(i)+'.png'),input1[i]+128)
            #save_imgs(os.path.join('./data/fashion_data/ori',str(i)+str(i)+'.png'),input2[i]+128)
            save_imgs(os.path.join(os.path.join(adv_save_train_path,str(args.nclass-1-int(index1[i]))),str(iter)+'_'+str(i)+'adv.png'),adv_1[i])
            save_imgs(os.path.join(os.path.join(process_save_train_path,str(args.nclass-1-int(index1[i]))),str(iter)+'_'+str(i)+'.png'),pro_1[i])
            save_imgs(os.path.join(os.path.join(adv_save_train_path,str(args.nclass-1-int(index2[i]))),str(iter)+'_'+str(i)+'_2adv.png'),adv_2[i])
            save_imgs(os.path.join(os.path.join(process_save_train_path,str(args.nclass-1-int(index2[i]))),str(iter)+'_'+str(i)+'_2.png'),pro_2[i])
            #if i>10:
            #    print(adv)
        #total_Rnoise += (float(torch.mean(abs(adv_p[0]-input1-128.)).cpu().detach()) + float(torch.mean(abs(adv_p[1]-input2-128.)).cpu().detach()))/2
        #total_noise += (float(torch.mean(abs(adv_g[0]-input1-128.)).cpu().detach()) + float(torch.mean(abs(adv_g[1]-input2-128.)).cpu().detach()))/2
        total_Rnoise +=  float(torch.mean(abs(adv_p[0]-input1-128.)).cpu().detach())
        total_noise += float(torch.mean(abs(adv_g[0]-input1-128.)).cpu().detach())
        if iter % args.print_freq == 0:
            if args.arch == 'en_decoder_finger':
                print(100*acc/((iter+1)*args.batch_size))
    print(total_Rnoise/iters)
    print(total_noise/iters)
    if args.arch == 'en_decoder_finger':
        print(100*acc/(iters*args.batch_size))
if __name__ == '__main__':
    main()
