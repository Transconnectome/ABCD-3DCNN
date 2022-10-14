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

class SIMPLE_CNN3D(nn.Module):
    def __init__(self,subject_data, args):
        super(SIMPLE_CNN3D,self).__init__()
        
        # theses variables are used for making fc layers
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target
        self.target = args.cat_target + args.num_target
        
        # defining and building layers
        in_channels = 1
        self.conv1 = nn.Conv3d(in_channels,8,kernel_size=3,stride=(1,1,1),padding=1) 
        self.conv2 = nn.Conv3d(8,16,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv3 = nn.Conv3d(16,32,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv4 = nn.Conv3d(32,64,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv5 = nn.Conv3d(64,128,kernel_size=3,stride=(1,1,1),padding=1)
        self.conv6 = nn.Conv3d(128,256,kernel_size=3,stride=(1,1,1),padding=1)
        
        self.batchnorm1 = nn.BatchNorm3d(8)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.batchnorm3 = nn.BatchNorm3d(64)
        
        self.classifiers = self._make_fclayers()

        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # 두가지 선택지, 1) padding = 0, kernel = 2; 2) padding = 1, kernel = 3 
        self.act = nn.ReLU()
        
           
    def _make_fclayers(self):
        FClayer = []

        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts()) 
            fc = nn.Sequential(nn.Linear(6**3*256,25),
                                    nn.Sigmoid(),
                                    nn.Dropout(),
                                    nn.Linear(25,self.out_dim),
                                    nn.Softmax(dim=1))
            
            FClayer.append(fc)
        
        for num_label in self.num_target:
            self.out_dim = 1
            fc = nn.Sequential(nn.Linear(6**3*256,25),
                                    nn.Sigmoid(),
                                    nn.Dropout(),
                                    nn.Linear(25,self.out_dim))
            FClayer.append(fc)
        
        return nn.ModuleList(FClayer) # must store list of multihead fc layer as nn.ModuleList to attach FC layer to cuda

        
    def forward(self,x):
        results = {}
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.batchnorm3(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
       
        x = x.view(x.size(0),-1)
        
        # passing through several fc layers with for loop
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](x)

        return  results 

def simple3D(subject_data, args):
    model = SIMPLE_CNN3D(subject_data, args)
    return model



