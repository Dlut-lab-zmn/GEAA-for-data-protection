import os
import numpy as np
import random
from PIL import Image
import torch
readed_images1 = []
readed_images2 = []
TR_index1 = []
TR_index2 = []
readed_images3 = []
readed_images4 = []
TR_index3 = []
TR_index4 = []
labels = []
labels2 = []
test_labels = []
test_labels2 = []
index4 = 0
index = 0
index2 = 0
index3 = 0
def one_hot(a,nclass):
    c = np.zeros(nclass)
    c[nclass - a - 1] = 1
    return c

def read_imgs(imgt):
		imgt = Image.open(imgt)
		imgt = np.array(imgt, np.float32, copy=False)
		if len(imgt.shape) == 2:
				imgt = np.expand_dims(imgt,2)
		imgt = torch.from_numpy(imgt-128.)
		imgt = imgt.transpose(0, 1).transpose(0, 2).contiguous()
		return imgt
def load_file_list(data_path,nclass = 10):
	global readed_images1;global TR_index1
	global labels

	print("load in!")
	path = os.path.join(data_path, "train/")
	images = []


	for i in range(nclass//2):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					images.append(directory+filename)
					labels.append(one_hot(i,nclass))

	for i in range(len(images)):
		imgt = read_imgs(images[i])
		readed_images1.append(torch.unsqueeze(imgt,0))
	print('Train load down',len(readed_images1))
	TR_index1 = np.arange(len(readed_images1))
	np.random.shuffle(TR_index1)
	return len(readed_images1)
def load_file_list2(data_path,nclass = 10):
	global readed_images2;global TR_index2
	global labels2
	print("load in!")
	images2 = []
	path = os.path.join(data_path,"train/")


	for i in range(nclass//2,nclass):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					images2.append(directory+filename)
					labels2.append(one_hot(i,nclass))
	for i in range(len(images2)):
		imgt = read_imgs(images2[i])
		readed_images2.append(torch.unsqueeze(imgt,0))
	print('Train load down',len(readed_images2))
	TR_index2 = np.arange(len(readed_images2))
	np.random.shuffle(TR_index2)
	return 	len(readed_images2)

def load_test_list(data_path,nclass = 10):
	global readed_images3;global TR_index3
	global test_labels
	path = os.path.join(data_path,'test/')
	test_images = []


	for i in range(nclass//2):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					test_images.append(directory+filename)
					test_labels.append(one_hot(i,nclass))
	for i in range(len(test_images)):
		imgt = read_imgs(test_images[i])
		readed_images3.append(torch.unsqueeze(imgt,0))
	print('Test load down',len(readed_images3))
	TR_index3 = np.arange(len(test_images))
	np.random.shuffle(TR_index3)
	return 	len(readed_images3)
def load_test_list2(data_path,nclass = 10):
	global readed_images4;global TR_index4
	global test_labels2
	path = os.path.join(data_path,'test/')
	test_images2 = []

	for i in range(nclass//2,nclass):
			directory = path + str(i) + '/'
			for filename in [y for y in os.listdir(directory)]:
					test_images2.append(directory+filename)
					test_labels2.append(one_hot(i,nclass))
	for i in range(len(test_images2)):
		imgt = read_imgs(test_images2[i])
		readed_images4.append(torch.unsqueeze(imgt,0))
	print('Test load down',len(readed_images4))
	TR_index4 = np.arange(len(readed_images4))
	np.random.shuffle(TR_index4)
	return 	len(readed_images4)

def get_batch(batch_size):
	global TR_index1;global readed_images1;global labels;global index
	Max_couter = len(readed_images1)
	Max_index = Max_couter//batch_size
	index = index%Max_index
	imgs =[];label=[]
	for q in TR_index1[index*batch_size:(index+1)*batch_size]:
				imgs.append(readed_images1[q])
				label.append(labels[q])
	imgs = torch.cat(imgs,0)
	index = (index+1)%Max_index
	if index == 0:
		random.shuffle(TR_index1)

	return imgs,label

def get_batch2(batch_size):
	global index2;global readed_images2;global labels2;global TR_index2
	Max_couter = len(readed_images2)
	Max_index = Max_couter//batch_size
	index2 = index2%Max_index
	imgs =[];label=[]

	for q in TR_index2[index2*batch_size:(index2+1)*batch_size]:
				imgs.append(readed_images2[q])
				label.append(labels2[q])
	imgs = torch.cat(imgs,0)
	index2 = (index2+1)%Max_index
	if index2 == 0:
		random.shuffle(TR_index2)

	return imgs,label


def get_test(batch_size):

	global index3;global readed_images3;global test_labels;global TR_index3
	Max_couter = len(readed_images3)
	Max_index = Max_couter//batch_size
	index3 = index3%Max_index
	imgs =[];label=[]

	for q in TR_index3[index3*batch_size:(index3+1)*batch_size]:
				imgs.append(readed_images3[q])
				label.append(test_labels[q])
	imgs = torch.cat(imgs,0)
	index3 = (index3+1)%Max_index
	if index3 == 0:
		random.shuffle(TR_index3)

	return imgs,label

def get_test2(batch_size):
	global index4;global readed_images4;global test_labels2;global TR_index4
	Max_couter = len(readed_images4)
	Max_index = Max_couter//batch_size
	index4 = index4%Max_index
	imgs =[];label=[]

	for q in TR_index4[index4*batch_size:(index4+1)*batch_size]:
				imgs.append(readed_images4[q])
				label.append(test_labels2[q])
	imgs = torch.cat(imgs,0)
	index4 = (index4+1)%Max_index
	if index4 == 0:
		random.shuffle(TR_index4)

	return imgs,label