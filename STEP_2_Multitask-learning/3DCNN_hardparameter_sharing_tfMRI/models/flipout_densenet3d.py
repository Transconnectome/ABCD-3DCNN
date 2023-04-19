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

# Bayesian (flipout) layer
# ref: https://github.com/IntelLabs/bayesian-torch
from bayesian_torch.layers import Conv3dFlipout
from bayesian_torch.layers import LinearFlipout


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
#(ref) pytorch-bayesian (github): https://github.com/IntelLabs/bayesian-torch
#(ref) pytorch-bayesian (model): https://github.com/IntelLabs/bayesian-torch/bayesian_torch/models/bayesian/resnet_flipout.py
class _DenseLayer(nn.Module):

    def __init__(self, 
                 num_input_features, 
                 growth_rate, 
                 bn_size, 
                 drop_rate,
                 #hyperparameters for flipout layer 
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0):
        super().__init__()

        ## DenseNet Composite function: BN -> relu -> 3x3 conv
        # 1
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', Conv3dFlipout(num_input_features, 
                                               bn_size * growth_rate, 
                                               kernel_size=1, 
                                               stride=1, 
                                               bias=False,
                                               prior_mean=prior_mean,
                                               prior_variance=prior_variance,
                                               posterior_mu_init=posterior_mu_init,
                                               posterior_rho_init=posterior_rho_init))
        

        # 2
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', Conv3dFlipout(bn_size * growth_rate, 
                                               growth_rate, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=1,
                                               bias=False,
                                               prior_mean=prior_mean,
                                               prior_variance=prior_variance,
                                               posterior_mu_init=posterior_mu_init,
                                               posterior_rho_init=posterior_rho_init))

        self.drop_rate = float(drop_rate)
        #self.memory_efficient = memory_efficient
    
    def bn_function(self, inputs) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output, kl = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output, kl

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x
        kl_sum = 0 

        bottleneck_output, kl = self.bn_function(prev_features)
        kl_sum += kl 
        new_features , kl = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        kl_sum += kl 
        if self.drop_rate > 0: 
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features, kl_sum  ## **

class _DenseBlock(nn.ModuleDict):
    # receive and concatenate the outputs of all previous blocks as inputs 
    # growth rate? the number of channel of feature map in each layer
    def __init__(self, 
                 num_layers, 
                 num_input_features, 
                 bn_size, 
                 growth_rate, 
                 drop_rate,
                 #hyperparameters for flipout layer 
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0):
        super().__init__()
        
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, 
                                bn_size, 
                                drop_rate,
                                prior_mean=prior_mean,
                                prior_variance=prior_variance,
                                posterior_mu_init=posterior_mu_init,
                                posterior_rho_init=posterior_rho_init)
            self.add_module("denselayer%d" % (i + 1), layer)
    
    def forward(self, init_features):
        kl_sum = 0
        features = [init_features]
        for name, layer in self.items():
            new_features, kl = layer(features)
            features.append(new_features)
            kl_sum += kl
        return torch.cat(features, 1), kl_sum

#class _Transition(nn.Sequential):
class _Transition(nn.Module):
    ## convolution + pooling between block
    # in paper: bach normalization -> 1x1 conv layer -> 2x2 average pooling layer

    def __init__(self, 
                 num_input_features, 
                 num_output_features, 
                 #hyperparameters for flipout layer   
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0):
        super().__init__()
        
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Conv3dFlipout(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False,
                                              prior_mean=prior_mean,
                                              prior_variance=prior_variance,
                                              posterior_mu_init=posterior_mu_init,
                                              posterior_rho_init=posterior_rho_init))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
        

    def forward(self, x): 
        trans_output, kl = self.conv(self.relu(self.norm(x)))
        trans_output = self.pool(trans_output)
        return trans_output, kl
        


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

    def __init__(self, subject_data, args,
                 n_input_channels=1,conv1_t_size=7,conv1_t_stride=2,no_max_pool=False,
                 growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,
                 bn_size=4,drop_rate=0, num_classes=1000,
                 #hyperparameters for flipout layer 
                 prior_mu=0.0,
                 prior_sigma=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0):

        super(DenseNet, self).__init__()
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target

        self.prior_mean = prior_mu
        self.prior_variance = prior_sigma
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init


        # First convolution 
        self.conv_stem = nn.Module() 
        self.conv_stem.add_module('conv0', Conv3dFlipout(n_input_channels,
                                num_init_features,
                                kernel_size=(conv1_t_size, 7, 7),
                                stride=(conv1_t_stride, 2, 2),
                                padding=(conv1_t_size // 2, 3, 3),
                                bias=False,
                                prior_mean=self.prior_mean,
                                prior_variance=self.prior_variance,
                                posterior_mu_init=self.posterior_mu_init,
                                posterior_rho_init=self.posterior_rho_init))
        self.conv_stem.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.conv_stem.add_module('relu0', nn.ReLU(inplace=True))
        self.conv_stem.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))


        # Each denseblock
        self.num_features = num_init_features
        self.block_config = block_config
        self.growth_rate = growth_rate
        self.conv_features = nn.ModuleDict()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=self.num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                prior_mean=self.prior_mean,
                                prior_variance=self.prior_variance,
                                posterior_mu_init=posterior_mu_init,
                                posterior_rho_init=posterior_rho_init)
            self.conv_features.add_module('denseblock{}'.format(i + 1), block)
            #self.conv_features.add_module('denseblock{}'.format(i + 1), block)
            self.num_features = self.num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=self.num_features,
                                    num_output_features=self.num_features // 2,
                                    prior_mean=self.prior_mean,
                                    prior_variance=self.prior_variance,
                                    posterior_mu_init=self.posterior_mu_init,
                                    posterior_rho_init=self.posterior_rho_init)
                self.conv_features.add_module('transition{}'.format(i + 1), trans)
                #self.conv_features.add_module('transition{}'.format(i + 1), trans)
                self.num_features = self.num_features // 2

        # Final batch norm
        self.add_module('norm', nn.BatchNorm3d(self.num_features))

        # Linear layer
        self.FClayers = self._make_fclayers()
        
        # init weights
        self.apply(self._weights_init)
    

    def _weights_init(self, m): 
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d): 
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear): 
            nn.init.constant_(m.bias, 0)

    

    def _make_fclayers(self):
        FClayer = []
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())
            FClayer.append(nn.Sequential(LinearFlipout(in_features=self.num_features,  
                                                       out_features=self.out_dim,                                   
                                                       prior_mean=self.prior_mean,
                                                       prior_variance=self.prior_variance,
                                                       posterior_mu_init=self.posterior_mu_init,
                                                       posterior_rho_init=self.posterior_rho_init)))

        for num_label in self.num_target:
            FClayer.append(nn.Sequential(LinearFlipout(in_features=self.num_features,  
                                                           out_features=1,                                   
                                                           prior_mean=self.prior_mean,
                                                           prior_variance=self.prior_variance,
                                                           posterior_mu_init=self.posterior_mu_init,
                                                           posterior_rho_init=self.posterior_rho_init)))

        return nn.ModuleList(FClayer)
    
    @torch.no_grad()
    def _hook_embeddings(self, x):
        features, _ = self.conv_features(x)
        for name, layer in self.conv_features.items(): 
            features, _ = layer(features)
        features = self.norm(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
        out = torch.flatten(out, 1)
        return out


    def forward(self, x):
        results = {}
        kl_sum = 0

        # stem layers 
        features, kl = self.conv_stem.conv0(x)
        kl_sum += kl 
        features = self.conv_stem.norm0(features)
        features = self.conv_stem.relu0(features)
        features = self.conv_stem.pool0(features)

        # dense layers 
        for name, layer in self.conv_features.items(): 
            features, kl = layer(features)
            kl_sum += kl
        features = self.norm(features)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1))
        features = torch.flatten(features, 1)

        # FC layers 
        for i in range(len(self.FClayers)):
            out, kl = self.FClayers[i](features)
            kl_sum += kl 
            if self.target[i] in self.cat_target: 
                results[self.target[i]] = F.softmax(out, dim=1)
            else: 
                results[self.target[i]] = out
        return results, kl_sum

def generate_model(model_depth, subject_data, args, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         drop_rate=0.2, 
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         drop_rate=0.2,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)
    return model

def densenet3D121(subject_data, args):
    model = generate_model(121, subject_data, args)
    return model

def densenet3D169(subject_data, args):
    model = generate_model(169, subject_data, args)
    return model

def densenet3D201(subject_data, args):
    model = generate_model(201, subject_data, args)
    return model

def densenet3D264(subject_data, args):
    model = generate_model(264, subject_data, args)
    return model
