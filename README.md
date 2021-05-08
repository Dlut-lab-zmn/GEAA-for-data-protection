# DAPA-for-data-protection
## This code is based on the paper 'Directional Adversarial Poisoning Attack (DAPA): First Adversarial Step towards the Shared Data Protection'

It supports:
- Adversarial poisoning attack
- Detoxification reconstruction
- Watermark reconstruction
- De-temperature
...

## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision)

### Prepare data.

We now support MNIST, Fashion-MNIST, CIFAR10, CIFAR100, SVHN.

### Start training

checkpoint notice:

model named as *cifar*.pth  *mnist*.pth or ....

The code could be seperated into several steps. It is very easy to re-implement our experiment results once you follow the following steps.

1. Train the classification network, e,g, cifar dataset and resnet18 model.
You can easy transfer to other classification network and dataset.
Download the data to ./cifar_data and copy the ./cifar_data/test/* to other dirs. 

- cd ./data
- python generate_dirs.py
- cp -r ./cifar_data/test/* ./adv/
- cp -r ./cifar_data/test/* ./process/

dirs example
.//
  /data//
      /cifar_data
         /cifar_data
            /train
                /0
                /1
                ...
            /test
                /0
                /1
                ...
         /adv
            /train
                /0
                /1
                ...
            /test
                /0
                /1
                ...
         /process
            /train
                /0
                /1
                ...
            /test
                /0
                /1
                ...
We also offer many exisiting models on the dir ./models for your convenience.
After you download the dataset and put it as the mentioned form, you can run the following code.

python class_net_train.py --load_path './data/cifar_data/cifar_data' --dataset 'cifar' --device '0' --save_name 'cifar'

The default model will decrease the learnning rate after 50 epochs, (0.1,0.01,0.001 totally 150 epochs.) , and saved at the dir './checkpoint/' as --save_name + '.pth'(such as 'cifar.pth').
