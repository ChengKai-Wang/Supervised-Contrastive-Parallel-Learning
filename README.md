# Supervised Contrastive Parallel Learning
## Setup
Tested under Python 3.8.12 in Ubuntu.
Install the required packages by
```
$ pip install -r requirements.txt
```

## File Description
* tiny-imagenet-200.zip：A zip file of tinyImageNet dataset processed under PyTorch ImageFolder format. Download link: https://drive.google.com/file/d/1R5QMeXAL_8XYqaDiGFFFoM1IwiJ5ZcBJ/view?usp=sharing

## Quick Start
### Unzip tiny-imagenet-200.zip
```
$ unzip tiny-imagenet-200.zip
```
### Config Settings
* data section
    * dataset: dataset for experiments. Options: cifar10, cifar100 or tinyImageNet 
    * train_batch_size: batch size for training
    * test_batch_size: batch size for testing
    * augmentation: use **basic** augmentation like BP commonly used, or **strong** augmentation like contrastive learning used. Options: basic, strong
* model section
    * model: model and loss functions for experiments. Options: CNN, CNN_AL, CNN_SCPL, CNN_PredSim, VGG, VGG_AL, VGG_SCPL, VGG_PredSim, resnet, resnet_AL, resnet_SCPL, resnet_PredSim
    * epochs: number of epochs for training
    * base_lr: initial learning rate
    * end_lr: learning rate at the end of training
* Example
```
[data]
dataset = tinyImageNet
train_batch_size = 128
test_batch_size = 1024
augmentation = strong


[model]
model = VGG
epochs = 200
base_lr = 0.001
end_lr = 0.00001
```
### Execute
```
$ python main.py
```
* Then, the tensorboard logger results will be saved in {work_dir}/{model}/{dataset}/tb_{i} folder where i means the i-th experiment