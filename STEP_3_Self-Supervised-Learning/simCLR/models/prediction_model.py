from utils.utils import set_backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class prediction_model(nn.Module):
    def __init__(self, subject_data, args):
        super(prediction_model, self).__init__()

        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target
        self.backbone, self.num_features = set_backbone(args)
        self.FClayers = self._make_fclayers()


    def _make_fclayers(self):
        FClayer = []
        
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())                        
            FClayer.append(nn.Sequential(nn.Linear(self.num_features, self.out_dim)))

        for num_label in self.num_target:
            FClayer.append(nn.Sequential(nn.Linear(self.num_features, 1)))

        return nn.ModuleList(FClayer)


    def forward(self, x):
        results = {}

        out = self.backbone(x)
        for i in range(len(self.FClayers)):
            results[self.target[i]] = self.FClayers[i](out)

        return results