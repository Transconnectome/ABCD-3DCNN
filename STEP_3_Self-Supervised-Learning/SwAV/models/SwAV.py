from doctest import OutputChecker
from re import L
from dataloaders.data_augmentation import applying_augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from utils.utils import set_backbone
import collections
from typing import List


class SwAV(nn.Module): 
    def __init__(self, args):
        super(SwAV, self).__init__()
        self.args = args 
        self.backbone, self.num_features = set_backbone(args)
        if isinstance(args.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(args.feat_dim, args.nmb_prototypes)
        elif args.nmb_prototypes > 0:
            self.prototypes = SinglePrototype(args.feat_dim, args.nmb_prototypes)
        self.l2norm = True

        self.projection_head = self._make_projection_head(self.num_features, args.feat_dim)


    def _make_projection_head(self, input_dim, output_dim):
        projcection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace = True),
            nn.Linear(input_dim, output_dim)
        )
        return projcection_head

    def forward_head(self, x):
        # forward projection_head    
        x = self.projection_head(x)

        # normalize output
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        # forward prototype codes
        return x, self.prototypes(x)
   
    def forward(self, inputs, mode=None):
        # inputs size = [tensor(batch, channel, x, y, z)] * n_views
        if not isinstance(inputs, list):
            inputs = [inputs]

        outputs = []
        for input in inputs:
                outputs.append(self.backbone(input.cuda()))            
        outputs = torch.cat(outputs)


        # forward projection head and prototypes
        embedding, prototype_output = self.forward_head(outputs)
        # embedding size = (batch_size*n_views, dim); prototype size = (batch_size * n_views, dim)
        return embedding, prototype_output


class MultiPrototypes(nn.Module): 
    def __init__(self, output_dim, nmb_prototypes: List):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))

        return out 


class SinglePrototype(nn.Module):
    def __init__(self, output_dim, nmb_prototypes: int):
        super(SinglePrototype, self).__init__()
        self.add_module("prototypes", nn.Linear(output_dim, nmb_prototypes, bias=False))

    def forward(self, x):
        out = getattr(self, "prototypes")(x) 
        return out    
    



@torch.no_grad() 
def distributed_sinkhorn(out, args):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    ######B = Q.shape[1] * args.world_size
    B = Q.shape[1]
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1 
    sum_Q = torch.sum(Q)
    ######dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K  
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        ######dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample myst be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment 
    return Q.t() 





