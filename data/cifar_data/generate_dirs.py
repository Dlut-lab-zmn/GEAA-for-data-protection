path = './'
import os
a = ['cifar_data/','process/','finger/','adv/']
b = ['train/','test/']
for i in range(len(a)):
    if not os.path.exists(path + a[i]):
          os.mkdir(path+a[i])
          os.mkdir(path+a[i]+b[0])
          os.mkdir(path+a[i]+b[1])
    for j in range(10):
          if not os.path.exists(path + a[i] + b[0] + str(j)):
              os.mkdir(path+a[i]+b[0]+str(j))
              os.mkdir(path+a[i]+b[1]+str(j))    