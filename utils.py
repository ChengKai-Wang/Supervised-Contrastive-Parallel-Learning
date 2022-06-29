import torch
import torch.nn as nn
import random
import numpy as np
import math
import torch.nn.functional as F

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R
class ALComponent(nn.Module):
    """
        Base class of a single associated learning block
        
        f: forward function
        g: autoencoder function
        b: bridge function
        inv: inverse function of autoencoder
        cb: creterion of bridge function
        ca: creterion of autoencoder
        cf: creterion of forward function
    """
    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        b: nn.Module,
        inv: nn.Module,
        cf: nn.Module,
        cb: nn.Module,
        ca: nn.Module
    )->None:
        super(ALComponent, self).__init__()
        self.f = f
        self.g = g
        self.b = b
        self.inv = inv
        self.cb = cb
        self.ca = ca
        self.cf = cf
    
    def forward(self, x=None, y=None, label=None):
        if self.training:
            s = self.f(x)
            #loss_f = 100 * self.cf(s, label)
            loss_f = 0
            s0 = self.b(s)
            t = self.g(y)
            t0 = self.inv(t)
            
            loss_b = self.cb(s0, t.detach()) # contrastive loss
            loss_ae = self.ca(t0, y)
            return s.detach(), t.detach(), loss_f, loss_b, loss_ae
        else:
            if y == None:
                s = self.f(x)
                return s
            else:
                t0 = self.inv(y)
                return t0
        
    # for bridge block inference
    def bridge_forward(self, x):
        s = self.f(x)
        s0 = self.b(s)
        t0 = self.inv(s0)
        return t0


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, input_neurons = 128, mid_neurons = 512, out_neurons = 1024, c_in = 256, shape = 32):
        super(ContrastiveLoss, self).__init__()
        input_neurons = int(c_in * shape * shape)
        if out_neurons == 0:
            self.linear = nn.Sequential(Flatten())
        elif mid_neurons == 0:
            self.linear = nn.Sequential(Flatten(), nn.Linear(input_neurons, out_neurons))
        else:
            self.linear = nn.Sequential(Flatten(), nn.Linear(input_neurons, mid_neurons), nn.ReLU(), nn.Linear(mid_neurons, out_neurons))
        self.temperature = temperature

    
    def forward(self, x, label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = self.linear(x)
        x =  nn.functional.normalize(x)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).float().to(device)
        denom_mask = torch.scatter(torch.ones_like(mask).to(device), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        logits = torch.div(torch.matmul(x, x.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()


        return loss

class PredSimLoss(nn.Module):
    def __init__(self, temperature = 0.1, input_neurons = 2048, c_in = 256, shape = 32):
        super().__init__()
        num_classes = 200
        self.conv_loss = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, bias=False)
        self.decoder_y = nn.Linear(input_neurons, num_classes)
        # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
        ks_h, ks_w = 1, 1
        dim_out_h, dim_out_w = shape, shape
        dim_in_decoder = c_in*dim_out_h*dim_out_w
        while dim_in_decoder > input_neurons and ks_h < shape:
            ks_h*=2
            dim_out_h = math.ceil(shape / ks_h)
            dim_in_decoder = c_in*dim_out_h*dim_out_w
            if dim_in_decoder > input_neurons:
               ks_w*=2
               dim_out_w = math.ceil(shape / ks_w)
               dim_in_decoder = c_in*dim_out_h*dim_out_w 
        if ks_h > 1 or ks_w > 1:
            pad_h = (ks_h * (dim_out_h - shape // ks_h)) // 2
            pad_w = (ks_w * (dim_out_w - shape // ks_w)) // 2
            self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(0, 0))
        else:
            self.avg_pool = None
    def forward(self, h, y):
        y_onehot = nn.functional.one_hot(y, num_classes=200).float()
        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)
        
        if self.avg_pool is not None:
            h = self.avg_pool(h)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))
        
        Ry = similarity_matrix(y_onehot).detach()
        loss_pred = (1-0.99) * F.cross_entropy(y_hat_local,  y.detach())
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        loss = loss_pred + loss_sim
        
        return loss
# for convolutional neural networks
def conv_layer_bn(in_channels: int, out_channels: int, activation: nn.Module, stride: int=1, bias: bool=False) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, bias = bias, padding = 1)
    bn = nn.BatchNorm2d(out_channels)
    if activation == None:
        return nn.Sequential(conv, bn)
    return nn.Sequential(conv, bn, activation)

def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(conv, bn)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
    
def accuracy(output, target):
    with torch.no_grad():
        bsz = target.shape[0]
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = correct[0].view(-1).float().sum(0, keepdim=True).mul_(100 / bsz)
        return acc
    
def add_noise_cifar(loader, noise_rate):
    """ 參考自 https://github.com/PaulAlbert31/LabelNoiseCorrection """
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_rate)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(10)))
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        if transform1 != None and transform2 == None:
            self.transform2 = transform1
        else:
            self.transform2 = transform2

    def __call__(self, x):
        if self.transform1 == None:
            return [x, x]
        return [self.transform1(x), self.transform2(x)]


