## =================================== ##
## ======= ResNet ======= ##
## =================================== ##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## ========= ResNet Model ========= #
#(ref)https://github.com/ML4HPC/Brain_fMRI ##Brain_fMRI/resnet3d.py
#(ref)https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3d, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups ## ***
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_3d(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3_3d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1_3d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    def __init__(self, block, layers, subject_data, args, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None): # parameters for hard parameter sharing model 
        
        super(ResNet3d, self).__init__()
        
        self.subject_data = subject_data

        # attribute for configuration
        self.layers = layers
        self.block_config = block
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target

        # attribute for building models
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        

        # building layers
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.FClayers = self._make_fclayers()        
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
  

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                            conv1x1_3d(self.inplanes, planes * block.expansion, stride),
                            norm_layer(planes * block.expansion),
                            )
            

        layers = []
        layers.append(self.block_config(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(self.block_config(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer))

        return layers, downsample


    def _make_fclayers(self):
        FClayer = []
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())                        
            FClayer.append(nn.Sequential(nn.Linear(512 * self.block_config.expansion, self.out_dim)))

        for num_label in self.num_target:
            FClayer.append(nn.Sequential(nn.Linear(512 * self.block_config.expansion, 1)))

        return nn.ModuleList(FClayer)


    def forward(self, x):
        results = {}

        # stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual blocks
        for segment, num_blocks in enumerate(self.layers):
            for b in range(num_blocks):
                # apply the residual skip out of _make_layers_
                residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                x = self.relu(residual + self.blocks[segment][b](x))


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](x)

        return results
                  


def resnet3D50(subject_data,args):
    layers = [3, 4, 6, 3]
    model = ResNet3d(Bottleneck3d, layers, subject_data, args)
    return model

def resnet3D101(subject_data,args):
    layers = [3, 4, 23, 3]
    model = ResNet3d(Bottleneck3d, layers, subject_data, args)
    return model
        
def resnet3D152(subject_data,args):
    layers = [3, 8, 36, 3]
    model = ResNet3d(Bottleneck3d, layers, subject_data, args)
    return model
      

## ====================================== ##
