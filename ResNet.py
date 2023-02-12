import torch
import torch.nn as nn
from utils import conv_layer_bn, Flatten, ALComponent, ContrastiveLoss, PredSimLoss



""" 修改自: https://github.com/batuhan3526/ResNet50_on_Cifar_100_Without_Transfer_Learning """
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv_layer_bn(in_channels, out_channels, nn.LeakyReLU(inplace=True), stride, False)
        self.conv2 = conv_layer_bn(out_channels, out_channels, None, 1, False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        if stride != 1:
            self.shortcut = conv_layer_bn(in_channels, out_channels, None, stride, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + self.shortcut(x))
        return out


class resnet18(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)

        self.conv2_x = self._make_layer(64, 64, [1, 1])
        self.conv3_x = self._make_layer(64, 128, [2, 1])
        self.conv4_x = self._make_layer(128, 256, [2, 1])
        self.conv5_x = self._make_layer(256, 512, [2, 1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.ce = nn.CrossEntropyLoss()


    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.training:
            return self.ce(output, y)
        else:
            return output
        
class ResNet_AL_Component(ALComponent):
    def __init__(self, conv: nn.Module, flatten_size: int = 1024, hidden_size: int = 500, out_features: int = 10):
        g_function = nn.Sigmoid() 
        b_function = nn.Sigmoid()

        f = conv
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features), g_function)

        cf = nn.Sequential()
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(ResNet_AL_Component, self).__init__(f, g, b, inv, cf, cb, ca)
         
class resnet18_AL(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        neurons = 500
        self.shape = 32
        
        conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)
        
        
        conv2_x = self._make_layer(64, 64, [1, 1])
        self.layer1 = ResNet_AL_Component(nn.Sequential(conv1, conv2_x), int(64 * self.shape * self.shape), neurons, num_classes)
        
        conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.layer2 = ResNet_AL_Component(conv3_x, int(128 * self.shape * self.shape), neurons, neurons)
        
        conv4_x = self._make_layer(128, 256, [2, 1])        
        self.shape /= 2
        self.layer3 = ResNet_AL_Component(conv4_x, int(256 * self.shape * self.shape), neurons, neurons)
        
        conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.layer4 = ResNet_AL_Component(conv5_x, int(512 * self.shape * self.shape), neurons, neurons)
        
    def forward(self, x, y):
        if self.training:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            # generate one-hot encoding
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
    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)
    
class resnet18_SCPL(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.shape = 32
        self.conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)

        self.conv2_x = self._make_layer(64, 64, [1, 1])
        self.sclLoss1 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 64, shape = self.shape)
        self.conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.sclLoss2 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 128, shape = self.shape)
        self.conv4_x = self._make_layer(128, 256, [2, 1])
        self.shape /= 2
        self.sclLoss3 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 256, shape = self.shape)
        self.conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.sclLoss4 = ContrastiveLoss(0.1, input_neurons = 2048, c_in = 512, shape = 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.ce = nn.CrossEntropyLoss()

    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x, y=None):
        loss = 0
        output = self.conv1(x)
        output = self.conv2_x(output)
        if self.training:
            loss += self.sclLoss1(output, y)
            output = output.detach()
        output = self.conv3_x(output)
        if self.training:
            loss += self.sclLoss2(output, y)
            output = output.detach()
        output = self.conv4_x(output)
        if self.training:
            loss += self.sclLoss3(output, y)
            output = output.detach()
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        if self.training:
            loss += self.sclLoss4(output, y)
            output = output.detach()
        
      
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output

        
class resnet18_PredSim(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()

        self.shape = 32
        self.conv1 = conv_layer_bn(3, 64, nn.LeakyReLU(inplace=True), 1, False)
        

        self.conv2_x = self._make_layer(64, 64, [1, 1])
        self.Loss1 = PredSimLoss(0.07, input_neurons = 2048, c_in = 64, shape = self.shape)
        self.conv3_x = self._make_layer(64, 128, [2, 1])
        self.shape /= 2
        self.Loss2 = PredSimLoss(0.07, input_neurons = 2048, c_in = 128, shape = self.shape)
        self.conv4_x = self._make_layer(128, 256, [2, 1])
        self.shape /= 2
        self.Loss3 = PredSimLoss(0.07, input_neurons = 2048, c_in = 256, shape = self.shape)
        self.conv5_x = self._make_layer(256, 512, [2, 1])
        self.shape /= 2
        self.Loss4 = PredSimLoss(0.07, input_neurons = 2048, c_in = 512, shape = self.shape)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.ce = nn.CrossEntropyLoss()

    def _make_layer(self, in_channels, out_channels, strides):
        layers = []
        cur_channels = in_channels
        for stride in strides:
            layers.append(BasicBlock(cur_channels, out_channels, stride))
            cur_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        loss = 0
        output = self.conv1(x)
        output = self.conv2_x(output)
        if self.training:
            loss += self.Loss1(output, y)
            output = output.detach()
        output = self.conv3_x(output)
        if self.training:
            loss += self.Loss2(output, y)
            output = output.detach()
        output = self.conv4_x(output)
        if self.training:
            loss += self.Loss3(output, y)
            output = output.detach()
        output = self.conv5_x(output)

        if self.training:
            loss += self.Loss4(output, y)
            output = output.detach()
        output = self.avg_pool(output)

        output = output.view(output.size(0), -1)

        output = self.fc(output)
        if self.training:
            loss += self.ce(output, y)
            return loss
        else:
            return output



