import torch
import torch.nn as nn
from utils import conv_layer_bn, Flatten, ALComponent, ContrastiveLoss, PredSimLoss

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        num_neurons = 300
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        
        self.conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        
        self.conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        
        
        self.fc = nn.Sequential(Flatten(), nn.Linear(4096, 5*num_neurons), nn.Sigmoid(), nn.Linear(5*num_neurons, num_classes))
        
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        
        out = self.fc(out)
        
        if self.training:
            return self.ce(out, y)
        else:
            return out.detach()
        
class CNN_AL_Component(ALComponent):
    def __init__(self, conv: nn.Module, flatten_size: int, hidden_size: int, out_features: int):

        g_function = nn.Sigmoid() 
        b_function = nn.Sigmoid()
        
        f = conv
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features))

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(CNN_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca)
        
class CNN_AL(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        neurons = 300

           
        conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        block_1 = nn.Sequential(conv1, conv2)
        
        self.layer1 = CNN_AL_Component(block_1, 32*32*32, neurons, num_classes)
        
        
        conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        block_2 = nn.Sequential(conv3, conv4)
        
        self.layer2 = CNN_AL_Component(block_2, 32*16*16, neurons, neurons)
        
        conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        block_3 = nn.Sequential(conv5, conv6)
        
        self.layer3 = CNN_AL_Component(block_3, 64*16*16, neurons, neurons)
        
        conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        block_4 = nn.Sequential(conv7, conv8)
        
        self.layer4 = CNN_AL_Component(block_4, 64*8*8, neurons, neurons)
    
    def forward(self, x, y):
        if self.training:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            y_onehot = torch.zeros([len(y), self.num_classes]).to(device)
            for i in range(len(y)):
                y_onehot[i][y[i]] = 1.
            
            _s = x
            _t = y_onehot
            total_loss = {'f':[], 'b':[],'ae':[]}

            
            _s, _t, loss_f, loss_b, loss_ae = self.layer1(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer2(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer3(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            
            _s, _t, loss_f, loss_b, loss_ae = self.layer4(_s, _t, y)
            total_loss['f'].append(loss_f)
            total_loss['b'].append(loss_b)
            total_loss['ae'].append(loss_ae)
            return total_loss
        else:
            _s = x
            _s = self.layer1(_s, None)
            _s = self.layer2(_s, None)
            _s = self.layer3(_s, None)
            _t0 = self.layer4.bridge_forward(_s)
            _t0 = self.layer3(None, _t0)
            _t0 = self.layer2(None, _t0)
            _t0 = self.layer1(None, _t0)
            return _t0
        
class CNN_SCPL(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        num_neurons = 300
        conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.layer1 = nn.Sequential(conv1, conv2)
        self.sclLoss1 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 32, shape = 32)
        
        conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        self.layer2 = nn.Sequential(conv3, conv4)
        self.sclLoss2 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 32, shape = 16)
        
        conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.layer3 = nn.Sequential(conv5, conv6)
        self.sclLoss3 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 64, shape = 16)
        
        conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        self.layer4 = nn.Sequential(conv7, conv8)
        self.sclLoss4 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 64, shape = 8)
        
        self.fc = nn.Sequential(Flatten(), nn.Linear(4096, 5*num_neurons), nn.Sigmoid(), nn.Linear(5*num_neurons, num_classes))
        
        self.ce = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        loss = 0
        output = self.layer1(x)
        if self.training:
            loss += self.sclLoss1(output, y)
            output = output.detach()
        output = self.layer2(output)
        if self.training:
            loss += self.sclLoss2(output, y)
            output = output.detach()
        output = self.layer3(output)
        if self.training:
            loss += self.sclLoss3(output, y)
            output = output.detach()
        output = self.layer4(output)
        if self.training:
            loss += self.sclLoss4(output, y)
            output = output.detach()

        
        
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output

class CNN_PredSim(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        num_neurons = 300
        conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.layer1 = nn.Sequential(conv1, conv2)
        self.Loss1 = PredSimLoss(0.1, input_neurons = 2048, c_in = 32, shape = 32)
        
        conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        self.layer2 = nn.Sequential(conv3, conv4)
        self.Loss2 = PredSimLoss(0.1, input_neurons = 2048, c_in = 32, shape = 16)
        
        conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        self.layer3 = nn.Sequential(conv5, conv6)
        self.Loss3 = PredSimLoss(0.1, input_neurons = 2048, c_in = 64, shape = 16)
        
        conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        self.layer4 = nn.Sequential(conv7, conv8)
        self.Loss4 = PredSimLoss(0.1, input_neurons = 2048, c_in = 64, shape = 8)
        
        self.fc = nn.Sequential(Flatten(), nn.Linear(4096, 5*num_neurons), nn.Sigmoid(), nn.Linear(5*num_neurons, num_classes))
        
        self.ce = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        loss = 0
        output = self.layer1(x)
        if self.training:
            loss += self.Loss1(output, y)
            output = output.detach()
        output = self.layer2(output)
        if self.training:
            loss += self.Loss2(output, y)
            output = output.detach()
        output = self.layer3(output)
        if self.training:
            loss += self.Loss3(output, y)
            output = output.detach()
        output = self.layer4(output)
        if self.training:
            loss += self.Loss4(output, y)
            output = output.detach()

        
        
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output