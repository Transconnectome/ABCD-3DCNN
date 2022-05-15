from utils.utils import set_backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections




class prediction_model(nn.Module):
    def __init__(self, subject_data, args):
        super(prediction_model, self).__init__()

        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target
        self.backbone, self.num_features = set_backbone(args)
        self.FClayers = self._make_fclayers()
        self.version = args.version


    def _make_fclayers(self):
        """
        In simCLR V1, there are 2 layers in projection head, and only (frozen) encoder is used to fine-tuning 
        In simCLR V2, there are 3 layers in projection head, and (frozen) encoder and the first (frozen) MLP layer is used to fine-tuning
        """
        FClayer = collections.OrderedDict([
            ('FClayers', nn.Linear(self.num_features, self.out_dim))
        ])
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())
            if self.version == 'simCLR_v1':          
                FClayer = collections.OrderedDict([
                    ('FClayers', nn.Linear(self.num_features, self.out_dim))
                    ])
            elif self.version == 'simCLR_v2':
                head1 = nn.Sequential(
                              nn.Linear(in_features=self.num_features, out_features=self.num_features),
                              nn.BatchNorm1d(self.num_features),
                              nn.ReLU()
                              )
                FClayer = collections.OrderedDict([
                    ('head1') , head1
                    ('FClayers', nn.Linear(self.num_features, self.out_dim))
                    ])


        for num_label in self.num_target:
            if self.version == 'simCLR_v1': 
                FClayer = collections.OrderedDict([
                    ('FClayer', nn.Linear(self.num_features, self.out_dim))
                    ])
            elif self.version == 'simCLR_v2':
                head1 = nn.Sequential(
                              nn.Linear(in_features=self.num_features, out_features=self.num_features),
                              nn.BatchNorm1d(self.num_features),
                              nn.ReLU()
                              )
                FClayer = collections.OrderedDict([
                    ('head1') , head1
                    ('FClayer', nn.Linear(self.num_features, self.out_dim))
                    ])
        return nn.Sequential(FClayer)


    def forward(self, x):
        results = {}

        out = self.backbone(x)
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](out)

        return results