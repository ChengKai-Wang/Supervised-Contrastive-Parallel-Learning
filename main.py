
import torch
from torchvision import transforms, datasets


import configparser

import os
import math
import sys
import time
import tensorboard_logger as tb_logger
from utils import AverageMeter, accuracy, TwoCropTransform

from ResNet import resnet18, resnet18_AL, resnet18_SCPL, resnet18_PredSim
from VGG import VGG, VGG_AL, VGG_SCPL, VGG_PredSim
from vanillaCNN import CNN, CNN_AL, CNN_SCPL, CNN_PredSim

def adjust_learning_rate(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    set_lr(optimizer, lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def read_config(path = "config.ini"):
    configs = dict()


    file = configparser.ConfigParser()
    file.read(path)
    dataset = file['data']['dataset']
    train_bsz = int(file['data']['train_batch_size'])
    test_bsz = int(file['data']['test_batch_size'])
    aug_type = file['data']['augmentation']
    model = file['model']['model']
    epochs = int(file['model']['epochs'])
    base_lr = float(file['model']['base_lr'])
    end_lr = float(file['model']['end_lr'])
    
    configs['dataset'] = dataset
    configs['train_bsz'] = train_bsz
    configs['test_bsz'] = test_bsz
    configs['aug_type'] = aug_type
    configs['model'] = model
    configs['epochs'] = epochs
    configs['base_lr'] = base_lr
    configs['end_lr'] = end_lr

    return configs

def set_loader(dataset, train_bsz, test_bsz, augmentation_type):
    if dataset == "cifar10":
        n_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "cifar100":
        n_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "tinyImageNet":
        n_classes = 200
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))
    
    if dataset == "cifar10" or dataset == "cifar100":
        normalize = transforms.Normalize(mean=mean, std=std)
        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if dataset == "tinyImageNet":
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])

        
        weak_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])

    if augmentation_type == "basic":
        source_transform = weak_transform
        target_transform = None
    elif augmentation_type == "strong":
        source_transform = TwoCropTransform(weak_transform, strong_transform)
        target_transform = TwoCropTransform(None)
    else:
        raise ValueError("Augmentation type not supported: {}".format(augmentation_type))


    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root='./cifar10', transform=source_transform, target_transform = target_transform,  download=True)
        test_set = datasets.CIFAR10(root='./cifar10', train=False, transform=test_transform)
    elif dataset == "cifar100":
        train_set = datasets.CIFAR100(root='./cifar100', transform=source_transform, target_transform = target_transform, download=True)
        test_set = datasets.CIFAR100(root='./cifar100', train=False, transform=test_transform)
    elif dataset == "tinyImageNet":
        train_set = datasets.ImageFolder('./tiny-imagenet-200/train', transform=source_transform)
        test_set = datasets.ImageFolder('./tiny-imagenet-200/val', transform=test_transform)




    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bsz, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsz, shuffle=False, pin_memory=True)
    

    return train_loader, test_loader, n_classes

def set_model(name, n_classes):
    if name == "VGG":
        model = VGG(n_classes)
    elif name == "VGG_AL":
        model = VGG_AL(n_classes)
    elif name == "VGG_SCPL":
        model = VGG_SCPL(n_classes)
    elif name == "VGG_PredSim":
        model = VGG_PredSim(n_classes)
    elif name == "resnet":
        model = resnet18(n_classes)
    elif name == "resnet_AL":
        model = resnet18_AL(n_classes)
    elif name == "resnet_SCPL":
        model = resnet18_SCPL(n_classes)
    elif name == "resnet_PredSim":
        model = resnet18_PredSim(n_classes)
    elif name == "CNN":
        model = CNN(n_classes)
    elif name == "CNN_AL":
        model = CNN_AL(n_classes)
    elif name == "CNN_SCPL":
        model = CNN_SCPL(n_classes)
    elif name == "CNN_PredSim":
        model = CNN_PredSim(n_classes)
    else:
        raise ValueError("Model not supported: {}".format(name))
    
    return model

def train(train_loader, model, optimizer, global_steps, epoch, aug_type, dataset):
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    base = time.time()
    for step, (X, Y) in enumerate(train_loader):
        if aug_type == "strong":
            if dataset == "cifar10" or dataset == "cifar100":
                X = torch.cat(X)
                Y = torch.cat(Y)
            else:
                X = torch.cat(X)
                Y = torch.cat([Y, Y])

        model.train()
        data_time.update(time.time()-base)

        if torch.cuda.is_available():
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
        bsz = Y.shape[0]

        global_steps += 1

        

        loss = model(X, Y)

        if type(loss) == dict:
            loss = sum(loss["f"]) + sum(loss["b"]) + sum(loss["ae"])
                            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)
        
        model.eval()
        with torch.no_grad():
            output = model(X, Y)
        acc = accuracy(output, Y)
        accs.update(acc.item(), bsz)

        batch_time.update(time.time()-base)
        base = time.time()

    # print info
    print("Epoch: {0}\t"
        "Time {1:.3f}\t"
        "DT {2:.3f}\t"
        "loss {3:.3f}\t"
        "acc {4:.3f}\t".format(epoch, (batch_time.avg)*len(train_loader), (data_time.avg)*len(train_loader), losses.avg, accs.avg))
    sys.stdout.flush()

    return losses.avg, accs.avg, global_steps



def test(test_loader, model, epoch):
    model.eval()

    batch_time = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        base = time.time()
        for step, (X, Y) in enumerate(test_loader):
 
            if torch.cuda.is_available():
                X = X.cuda(non_blocking=True)
                Y = Y.cuda(non_blocking=True)
            bsz = Y.shape[0]

            output = model(X, Y)

            acc = accuracy(output, Y)
            accs.update(acc.item(), bsz)

            batch_time.update(time.time()-base)
            base = time.time()

    # print info

    print("Epoch: {0}\t"
        "Time {1:.3f}\t"
        "Acc {2:.3f}\t".format(epoch, batch_time.avg*len(test_loader), accs.avg))
    
    print("================================================")
    sys.stdout.flush()

    return accs.avg



def main(i):
    best_acc = 0
    configs = read_config()
    train_loader, test_loader, n_classes = set_loader(configs['dataset'], configs['train_bsz'], configs['test_bsz'], configs['aug_type'])
    configs['max_steps'] = configs['epochs'] * len(train_loader)
    model = set_model(configs['model'], n_classes).cuda() if torch.cuda.is_available() else set_model(configs['model'], n_classes)



    # optimizer = torch.optim.SGD(model.parameters(), lr=configs['base_lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['base_lr'])

    logger = tb_logger.Logger("./logger/" + configs['model'] + "-" + configs['dataset'] + "-tb_{0}".format(i), flush_secs=2)
    global_steps = 0
    for epoch in range(1, configs['epochs'] + 1):
        lr = adjust_learning_rate(optimizer, configs['base_lr'], configs['end_lr'], global_steps, configs['max_steps'])
        
        print("lr: {:.6f}".format(lr))
        loss, train_acc, global_steps = train(train_loader, model, optimizer, global_steps, epoch, configs['aug_type'], configs['dataset'])

        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', lr, epoch)

        test_acc = test(test_loader, model, epoch)
        logger.log_value('test_acc', test_acc, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
        
    state = {
        "configs": configs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    save_files = os.path.join("./save_models/", "ckpt_last_{0}.pth".format(i))
    torch.save(state, save_files)

    del state
    print("Best accuracy: {:.2f}".format(best_acc))

if __name__ == '__main__':
    n_trials = 5
    for i in range(n_trials):
        main(i)









