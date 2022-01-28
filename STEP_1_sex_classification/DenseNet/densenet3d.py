## =================================== ##
## ======= DensNet ======= ##
## =================================== ##

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from torch import optim
from torch import Tensor
# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import collections
import numpy as np
#from torchsummary import summary
import time
import copy

## ========= DenseNet Model ========= #
#(ref) explanation - https://wingnim.tistory.com/39
#(ref) densenet3d - https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
#(ref) pytorch - https://pytorch.org/vision/0.8/_modules/torchvision/models/densenet.html

class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()

    ## DenseNet Composite function: BN -> relu -> 3x3 conv
    # 1
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))

        # 2
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))

        self.drop_rate = float(drop_rate)
        #self.memory_efficient = memory_efficient
    
    def bn_function(self, inputs) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return new_features  ## **

class _DenseBlock(nn.ModuleDict):
    # receive and concatenate the outputs of all previous blocks as inputs 
    # growth rate? the number of channel of feature map in each layer
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
    
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    ## convolution + pooling between block
    # in paper: bach normalization -> 1x1 conv layer -> 2x2 average pooling layer

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=1,conv1_t_size=7,conv1_t_stride=1,no_max_pool=False,
                 growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,
                 bn_size=4,drop_rate=0,num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = [('conv1',nn.Conv3d(n_input_channels,
                                            num_init_features,
                                            kernel_size=(conv1_t_size, 7, 7),
                                            stride=(conv1_t_stride, 2, 2),
                                            padding=(conv1_t_size // 2, 3, 3),
                                            bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(collections.OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            print(num_features)
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1) # **
        out = self.classifier(out)
        return out

def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    return model

def densenet121():
    model = generate_model(121)
    print(model)
    return model

"""
def densenet121(pretrained: bool = False, progress: bool = True, **kwargs):
    return DenseNet("densenet121", 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)

def densenet161(pretrained: bool = False, progress: bool = True, **kwargs):
    return DenseNet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)

def densenet169(pretrained: bool = False, progress: bool = True, **kwargs):
    return DenseNet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)

def densenet201(pretrained: bool = False, progress: bool = True, **kwargs):
    return DenseNet("densenet201", 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)
"""
