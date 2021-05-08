
import torchvision.transforms as transforms
import os
import numpy as np
import random
import torch
from PIL import Image


readed_TRimages = []
readed_TEimages = []
labels = []

test_labels = []
TR_index = []
TE_index = []
index1 = 0
index = 0

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def one_hot(a,nclass):
    c = np.zeros(nclass)
    c[nclass - a - 1] = 1
    return c


def read_imgs(imgt,mode):
		if 'cifar' in imgt or 'imagenet' in imgt or 'svhn' in imgt:
				imgt = Image.open(imgt)
				if mode == 'train':
						imgt = cifar_transform_train(imgt)
				if mode == 'test':
						imgt = cifar_transform_test(imgt)
		elif 'mnist' in imgt or 'fashion' in imgt:
				imgt = Image.open(imgt)
				imgt = mnist_transform(imgt)
		else:
				print('this dataset is not available')
				assert imgt is tuple
		return imgt
def load_file_list(data_path,nclass = 10):
	global readed_TRimages
	global labels;global TR_index
	print("load in!")
	images = []
	path = os.path.join(data_path, "train/")
	
	for i in range(nclass):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					images.append(directory+filename)
					labels.append(one_hot(i,nclass))

	for i in range(len(images)):
		imgt = read_imgs(images[i],'train')
		readed_TRimages.append(torch.unsqueeze(imgt,0))
	print('Train load down',len(readed_TRimages))
	TR_index = np.arange(len(images))
	np.random.shuffle(TR_index)
	return len(images)


def load_test_list(data_path,nclass = 10):
	global readed_TEimages
	global test_labels;global TE_index
	path = os.path.join(data_path, "test/")
	test_images = []
	for i in range(nclass):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					test_images.append(directory+filename)
					test_labels.append(one_hot(i,nclass))
	for i in range(len(test_images)):
		imgt = read_imgs(test_images[i],'train')
		readed_TEimages.append(torch.unsqueeze(imgt,0))
	print('Test load down',len(readed_TEimages))
	TE_index = np.arange(len(test_images))
	return 	len(test_images)


def get_batch(batch_size,gray = False):
	global index;global readed_TRimages;global labels;global TR_index
	Max_couter = len(readed_TRimages)
	Max_index = Max_couter//batch_size
	index = index%Max_index
	imgs =[];label=[]

	for q in TR_index[index*batch_size:(index+1)*batch_size]:
				imgs.append(readed_TRimages[q])
				label.append(labels[q])
	imgs = torch.cat(imgs,0)
	index = (index+1)%Max_index
	if index == 0:
		random.shuffle(TR_index)
	return imgs,label


def get_test(batch_size,gray = False):
	global index1;global readed_TEimages;global test_labels;global TE_index
	Max_couter = len(readed_TEimages)
	Max_index = Max_couter//batch_size
	index1 = index1%Max_index
	imgs =[];label=[]

	for q in TE_index[index1*batch_size:(index1+1)*batch_size]:
				imgs.append(readed_TEimages[q])
				label.append(test_labels[q])
	imgs = torch.cat(imgs,0)
	index1 = (index1+1)%Max_index
	if index1 == 0:
		random.shuffle(TE_index)
	return imgs,label






