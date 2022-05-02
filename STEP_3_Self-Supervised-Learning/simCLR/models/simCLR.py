import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import set_backbone


class simCLR(nn.Module):
    def __init__(self, args, embedding_size=512):
        super(simCLR, self).__init__()
        self.backbone, self.num_features = set_backbone(args)
        self.embedding_size = embedding_size
        self.projection_head = self._projection_head()


    def _projection_head(self):
        projection_head = nn.Sequential(
            nn.Linear(in_features=self.num_features, out_features=self.num_features),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(),
            nn.Linear(in_features=self.num_features, out_features=self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
            )
        return projection_head
                                                    

    def forward(self, x):
        embedding = self.backbone(x)
        embedding = self.projection_head(embedding)
        
        return embedding