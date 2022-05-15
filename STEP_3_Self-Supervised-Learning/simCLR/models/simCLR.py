import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import set_backbone
import collections

class simCLR(nn.Module):
    def __init__(self, args, embedding_size=512):
        super(simCLR, self).__init__()
        self.backbone, self.num_features = set_backbone(args)
        self.embedding_size = embedding_size
        self.projection_head = self._projection_head()
        self.version = args.version


    def _projection_head(self):
        """
        In simCLR V1, there are 2 layers in projection head, and only (frozen) encoder is used to fine-tuning 
        In simCLR V2, there are 3 layers in projection head, and (frozen) encoder and the first (frozen) MLP layer is used to fine-tuning
        """
        head1 = nn.Sequential(
                              nn.Linear(in_features=self.num_features, out_features=self.num_features),
                              nn.BatchNorm1d(self.num_features),
                              nn.ReLU()
                              )
        

        head2 = nn.Sequential(
                              nn.Linear(in_features=self.num_features, out_features=self.num_features),
                              nn.BatchNorm1d(self.num_features),
                              nn.ReLU()
                              )
        

        FClayer = nn.Linear(in_features=self.num_features, out_features=self.embedding_size)

        if self.version == 'simCLR_v1':
            projection_head = nn.Sequential(collections.OrderedDict([
                ('head1',head1),
                ('FClayer', FClayer)
            ]))
        elif self.version == 'simCLR_v2':            
            projection_head = nn.Sequential(collections.OrderedDict([
                ('head1',head1),
                ('head2',head2),
                ('FClayer', FClayer)
            ]))
        
        return projection_head
                                                    

    def forward(self, x):
        embedding = self.backbone(x)
        embedding = self.projection_head(embedding)
        
        return embedding