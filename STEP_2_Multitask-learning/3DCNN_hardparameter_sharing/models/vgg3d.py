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

# model definition
class VGG3D(nn.Module):
    def __init__(self, model_code, subject_data, args):
        super(VGG3D,self).__init__()
        
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target
        self.target = args.cat_target + args.num_target
        
        self.layers = self._make_layers(model_code,in_channels=1)
        self.FClayers = self._make_fclayers()

        
    def _make_layers(self, model_code,in_channels):
        layers = []

        for x in model_code:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))]
            else:
                layers += [nn.Conv3d(in_channels=in_channels,
                                    out_channels=x,
                                    kernel_size=3,
                                    stride=(1,1,1),
                                    padding=1)]
                layers += [nn.BatchNorm3d(x)]
                layers += [nn.ReLU()]
                in_channels = x

        return nn.Sequential(*layers)

    
    def _make_fclayers(self):
        FClayer = []

        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts()) 
            fc = nn.Sequential(nn.Linear(3**3*512,4096),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(4096,25),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(25,self.out_dim),
                                    nn.Softmax(dim=1))
            
            FClayer.append(fc)
        
        for num_label in self.num_target:
            self.out_dim = 1
            fc = nn.Sequential(nn.Linear(3**3*512,4096),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(4096,25),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(25,self.out_dim))
             
            FClayer.append(fc)
        
        return nn.ModuleList(FClayers) # must store list of multihead fc layer as nn.ModuleList to attach FC layer to cuda
    
    
    def forward(self,x):
        results = {}
        
        x = self.layers(x)
        x = x.view(x.size(0),-1)

        # passing through several fc layers with for loop
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](x)

        return  results 



def vgg3D11(subject_data,args):
    model_code = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
    model = VGG3D(model_code, subject_data, args)
    return model

def vgg3D13(subject_data,args):
    model_code = [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']
    model = VGG3D(model_code, subject_data, args)
    return model

def vgg3D16(subject_data,args):
    model_code = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
    model = VGG3D(model_code, subject_data, args)
    return model

def vgg3D19(subject_data,args):
    model_code = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
    model = VGG3D(model_code, subject_data, args)
    return model
    

