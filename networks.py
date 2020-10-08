from torch import nn
import torch
import os
import numpy as np
from torch.autograd import grad as torch_grad
from utils import mkdir

class ConvenientModel(nn.Module):
    def __init__(self, path):
        super(ConvenientModel, self).__init__()
        self.path = path

    def save(self, global_step):
        mkdir(self.path)
        torch.save({'step': global_step, 'model_state_dict': self.state_dict()}, self.path+'{:010d}'.format(global_step))

    def load(self, checkpoint=None):
        mkdir(self.path)
        if checkpoint:
            path = self.path + checkpoint
        else:
            f = os.listdir(self.path)
            if f:
                path = self.path + sorted(f)[-1]
            else:
                print('No save found')
                return 0
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded. Step', checkpoint['step'])
        return checkpoint['step']

    def gradient(self, x, get_scalar=False):
        '''
        :param x: The input image, either as np array or as torch tensor. Shape (batch, ch, x, y)
        :return: The gradient of the regulariser wrt the input. Same shape and type as input.
        '''
        if type(x) == np.ndarray:
            inp = torch.tensor(x, requires_grad=True).cuda()
            from_np = True
        else:
            inp = x.clone().requires_grad_(True)
            from_np = False
        p = self(inp)
        gradients = torch_grad(outputs=p, inputs=inp,
           grad_outputs=p.new_ones(p.size()),
           create_graph=False, retain_graph=False)[0]
        if from_np:
            gradients = gradients.detach().cpu().numpy()
            p = p.detach().cpu().numpy()
        if get_scalar:
            return gradients, p
        else:
            return gradients

    def forward(self, x):
        pass


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=[3, 3], padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=.1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], padding=0)
        if downsample:
            self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        res = x + skip
        if self.downsample:
            res = self.pool(res)
        return res


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ConvBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=.1)
        if downsample:
            self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        if self.downsample:
            x = self.pool(x)
        return x


class ResNet(ConvenientModel):
    name = 'ResNets'
    blocks = ResBlock
    def __init__(self, channels, downsamples, dense_neurons=128, exp_name='1', base_path='/store/CCIMI/sl767/Experiments/'):
        name = '_'.join([f'{x}_{int(y)}' for x, y in zip(channels[1:], downsamples)])+f'_Neurons_{dense_neurons}'
        path = os.path.join(base_path, f'{exp_name}/{self.name}/{name}/')
        super(ResNet, self).__init__(path)
        layers = []
        for c_in, c_out, d in zip(channels[:-1], channels[1:], downsamples):
            layers.append(self.blocks(c_in, c_out, d))
        self.feature_ext = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=channels[-1], out_features=dense_neurons),
            nn.LeakyReLU(negative_slope=.1),
            nn.Linear(in_features=dense_neurons, out_features=1)
        )
        self.load()


    def forward(self, x):
        r = self.feature_ext(x)
        mean = r.mean([2, 3])
        return self.classifier(mean)

class ConvNet(ResNet):
    name = 'ConvNets'
    blocks = ConvBlock
