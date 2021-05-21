# GAPA-for-data-protection
## This code is based on the paper 'Guided Adversarial Poisoning Attack (GAPA): First Adversarial Step towards the Shared Data Protection'

It supports:
- Adversarial poisoning attack
- Detoxification reconstruction
- Watermark reconstruction
- De-temperature
...

## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision)
- Opencv

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
- cp -r ./cifar_data/test/* ./adv/test
- cp -r ./cifar_data/test/* ./process/test

We also offer many exisiting models on the dir ./models for user's convenience.
After you download the dataset and put it follow the example form, you can run the following code.
```bash
$ python class_net_train.py --load_path './data/cifar_data/cifar_data' --dataset 'cifar' --device '0' --save_name 'cifar'
```
The default model will decrease the learnning rate after 50 epochs, (0.1,0.01,0.001 totally 150 epochs.) , and saved at the dir './checkpoint/' as --save_name + '.pth'(such as 'cifar.pth').

2. Train the GAPA model.
```bash
$ python train_en_decoder_finger.py --dataset 'cifar' --gama 0.9 --bound 10 --device '0'
```
The model will be saved in dir 'en_decoder_finger' as {str(epoch_iter)+'_'+ --dataset+'.tar'}, such as 14_cifar.tar.

3. Generate the poisoned data and detoxified data.
```bash
$ python test_en_decoder_finger.py  --dataset 'cifar' --resume './en_decoder_finger/14_cifar.tar' '--device '0'
```
The poisoned data is stored in './data/cifar_data/adv/train' and the detoxified data is stored in './data/cifar_data/process/train'

4. Train the classification model using the generated data.
```bash
$ python class_net_train.py --load_path './data/cifar_data/adv/' --device '0'
$ python class_net_train.py --load_path './data/cifar_data/process/' --device '0'
```
