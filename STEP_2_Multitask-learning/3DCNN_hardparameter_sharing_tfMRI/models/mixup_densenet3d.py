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

from densenet3d import DenseNet



class DenseNet_Mixup(DenseNet):
    def __init__(self, subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs):
        super().__init__(subject_data=subject_data, args=args, 
                         num_init_features=num_init_features, 
                         growth_rate=growth_rate, 
                         block_config=block_config,
                         **kwargs)

        assert len(self.FClayers) == 1 

    def forward(self, x, lam=None, index=None):
        results = {}
        # stem layers
        features = self.conv_stem.conv0(x)
        features = self.conv_stem.norm0(features)
        features = self.conv_stem.relu0(features)
        features = self.conv_stem.pool0(features)

        # dense layers 
        for name, layer in self.conv_features.items(): 
            features = layer(features)
        features = self.norm(features)

        # mixup process for manifold mixup
        if (lam is not None) and (index is not None): 
            """
            ref: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/models/wide_resnet.py
            """
            # mixup process 
            features = lam * features + (1 - lam) * features[index]

        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1))
        features = torch.flatten(features, 1)
        # FC layers 
        for i in range(len(self.FClayers)):
            out = self.FClayers[i](features)
            if self.target[i] in self.cat_target:
                results[self.target[i]] = F.softmax(out,dim=1)
            else: 
                results[self.target[i]] = out

        return results


def generate_model(model_depth, subject_data, args, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet_Mixup(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet_Mixup(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet_Mixup(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet_Mixup(subject_data, args,
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
