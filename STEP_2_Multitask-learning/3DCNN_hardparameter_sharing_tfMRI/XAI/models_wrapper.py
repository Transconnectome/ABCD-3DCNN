import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os 
 
# setting path
sys.path.append(os.path.dirname(os.getcwd()))
from models.densenet3d import DenseNet #model script

class DenseNet_IntegratedGrad(DenseNet):
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

    def forward(self, x):
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
        out = torch.flatten(out, 1)

        
        out = self.FClayers[0](out)
        """
        # stem layers
        features = self.conv_stem.conv0(x)
        features = self.conv_stem.norm0(features)
        features = self.conv_stem.relu0(features)
        features = self.conv_stem.pool0(features)
        # dense layers 
        for name, layer in self.conv_features.items(): 
            features = layer(features)
        features = self.norm(features)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1))
        features = torch.flatten(features, 1)
        out = self.FClayers[0](features)
        if self.cat_target: 
            out = F.softmax(out,dim=1)
        elif self.num_target: 
            pass 
        return out


def generate_model(model_depth, subject_data, args, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet_IntegratedGrad(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet_IntegratedGrad(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet_IntegratedGrad(subject_data, args,
                         num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet_IntegratedGrad(subject_data, args,
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
