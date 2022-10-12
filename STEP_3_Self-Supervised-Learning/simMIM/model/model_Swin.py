# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Video Swin: https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/mmaction/models/backbones/swin_transformer.py#L12 
# --------------------------------------------------------
import math
import logging
from functools import partial, reduce, lru_cache
from operator import mul
from os import pread
from einops import rearrange
from collections import OrderedDict
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.runner import load_checkpoint

from .swin_transformer import PatchEmbed3D, PatchMerging3D, BasicLayer
from .layers.helpers import to_3tuple

class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 simMIM_pretrained=False, 
                 patch_size=4,
                 num_classes=1,
                 in_channels=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.simMIM_pretrained = simMIM_pretrained
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size = to_3tuple(window_size)
        self.patch_size = to_3tuple(patch_size)
        self.in_channels = in_channels
        self.num_classes = num_classes

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop : nn.Module = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # the last norm layer
        self.norm = norm_layer(self.num_features)

        # the last pooling layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # predition head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self._freeze_stages()
        self.init_weights()
    
    @torch.jit.ignore
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.jit.ignore
    #def inflate_weights(self, logger):
    def inflate_pos_emb(self, state_dict):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """

        # bicubic interpolate relative_position_bias_table if not match
        for l, layer in enumerate(self.layers):
            for b, block in enumerate(self.layers[l].blocks):
                relative_position_bias_table_pretrained =  state_dict['layers.%s.blocks.%s.attn.relative_position_bias_table' % (str(l), str(b))]
                relative_position_bias_table_current = self.layers[l].blocks[b].attn.relative_position_bias_table.data 
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
                wd = self.window_size[0]
                if nH1 != nH2:
                    print(f'Error in loading layers.{l}.blocks.{b}.attn.relative_position_bias_table, passing')
                else: 
                    if L1 == L2:
                        S1 = int(L1 ** 0.5)
                        relative_position_bias_table_pretrained_resize = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1,0).view(1,nH1, S1, S1), size=(2*self.winodw_size[1]-1, 2*self.window_size[2]-1),
                            mode='bicubic')
                        relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resize.view(nH2, L2).permute(1,0)
                setattr(self, 'layers.%i.blocks.%i.attn.relative_position_bias_table.data' % (l, b), relative_position_bias_table_pretrained.repeat(2*wd-1, 1)) 


        #msg = self.load_state_dict(state_dict, strict=False)
        #logger.info(msg)
        #logger.info(f"=> loaded successfully '{self.pretrained}'")
        

    @torch.jit.ignore
    def load_simMIM_pretrained(self):
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        prefix = 'encoder'

        # load patch embedding layers 
        setattr(self, 'patch_embed.proj.weight.data', state_dict['%s.patch_embed.proj.weight' % prefix])
        setattr(self, 'patch_embed.proj.bias.data', state_dict['%s.patch_embed.proj.bias' % prefix])

        # load attention layers
        for l, layer in enumerate(self.layers):
            for b, block in enumerate(self.layers[l].blocks): 
                # initial norm layer
                setattr(self, 'layers[%s].blocks[%s].norm1.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.norm1.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].norm1.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.norm1.bias' % (prefix, str(l), str(b))])
                # attention qkv parameters 
                setattr(self, 'layers[%s].blocks[%s].attn.qkv.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.attn.qkv.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].attn.qkv.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.attn.qkv.bias' %(prefix, str(l), str(b))])
                # attention projection layer
                setattr(self, 'layers[%s].blocks[%s].attn.proj.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.attn.proj.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].attn.proj.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.attn.proj.bias' % (prefix, str(l), str(b))])
                # attention relative position bias. Do not need to load relative_position_index  
                setattr(self, 'layers[%s].blocks[%s].attn.relative_position_bias_table.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.attn.relative_position_bias_table' % (prefix, str(l), str(b))])
                # last norm layer
                setattr(self, 'layers[%s].blocks[%s].norm2.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.norm2.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].norm2.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.norm2.bias' % (prefix, str(l), str(b))])
                # fc layer 
                setattr(self, 'layers[%s].blocks[%s].mlp.fc1.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.mlp.fc1.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc1.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.mlp.fc1.bias' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc2.weight.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.mlp.fc2.weight' % (prefix, str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc2.bias.data' % (str(l), str(b)), state_dict['%s.layers.%s.blocks.%s.mlp.fc2.bias' % (prefix, str(l), str(b))])
            
            # merging layer 
            if l < len(self.layers) - 1:
                setattr(self, 'layers[%s].downsample.reduction.weight.data' % str(l), state_dict['%s.layers.%s.downsample.reduction.weight' % (prefix, str(l))])
                setattr(self, 'layers[%s].downsample.norm.weight.data' % str(l), state_dict['%s.layers.%s.downsample.norm.weight' % (prefix, str(l))])
                setattr(self, 'layers[%s].downsample.norm.bias.data' % str(l), state_dict['%s.layers.%s.downsample.norm.bias' % (prefix, str(l))])
        
        del checkpoint
        del state_dict
        torch.cuda.empty_cache()        
        print(f"=> loaded successfully '{self.pretrained}'")

    @torch.jit.ignore
    def load_pretrained2d(self): 
        """ Patch embedding and Patch merging layers couldn't be inflate 2D -> 3D 
        ImageNet pretrained model has 3 channels for patch embedding, but this model only handle 1 channel patch emebdding. 
        ImageNet pretrained model merging Height, and Width of images, but this model additionally merging depth. Thus hidden dimension of merging layer doesn't match. 
        (In Video Transformer merging layer only merging Height and Width)
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # inflate positional embedding 2D -> 3D 
        self.inflate_pos_emb(state_dict)

        # load attetion layers 
        for l, layer in enumerate(self.layers):
            for b, block in enumerate(self.layers[l].blocks):
                # initial norm layer
                setattr(self, 'layers[%s].blocks[%s].norm1.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.norm1.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].norm1.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.norm1.bias' % (str(l), str(b))])
                # attention qkv parameters 
                setattr(self, 'layers[%s].blocks[%s].attn.qkv.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.attn.qkv.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].attn.qkv.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.attn.qkv.bias' %(str(l), str(b))])
                # attention projection layer
                setattr(self, 'layers[%s].blocks[%s].attn.proj.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.attn.proj.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].attn.proj.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.attn.proj.bias' % (str(l), str(b))])
                # last norm layer
                setattr(self, 'layers[%s].blocks[%s].norm2.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.norm2.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].norm2.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.norm2.bias' % (str(l), str(b))])
                # fc layer 
                setattr(self, 'layers[%s].blocks[%s].mlp.fc1.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.mlp.fc1.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc1.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.mlp.fc1.bias' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc2.weight.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.mlp.fc2.weight' % (str(l), str(b))])
                setattr(self, 'layers[%s].blocks[%s].mlp.fc2.bias.data' % (str(l), str(b)), state_dict['layers.%s.blocks.%s.mlp.fc2.bias' % (str(l), str(b))])

        del checkpoint
        del state_dict
        torch.cuda.empty_cache()
        print(f"=> loaded successfully '{self.pretrained}'")

    @torch.jit.ignore
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            #logger = get_root_logger()
            #logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                #self.inflate_weights(logger)
                #self.inflate_pos_emb()
                self.load_pretrained2d()
            else:
                if self.simMIM_pretrained: 
                    # load 3D model pretrained by simMIM
                    self.load_simMIM_pretrained()
                
                else:
                    # load 3D model 
                    state_dict = torch.load(self.pretrained, map_location='cpu')
                    self.load_state_dict(state_dict['model'])
                #load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def forward(self, x: torch.Tensor):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())
        
        #x = rearrange(x, 'b c d h w -> b (d h w) c') # B L C    # original version
        ## torchscript version
        x = torch.einsum('bcdhw -> bdhwc', x) 
        B, D, H, W, C = x.size() 
        x = x.reshape(B, D * H * W, C)  # B L C 
        ######################   
        x = self.norm(x)  # B L C   
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


def swin_small_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=96, num_heads=[3, 6, 12, 24],              
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def swin_base_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32],                 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#def swin_base_patch4_window8_3D(pretrained, pretrained2d, simMIM_pretrained, window_size, drop_rate, num_classes):
#    model = torch.jit.script(SwinTransformer3D(
#        patch_size=4, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32],                 
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
#        pretrained=pretrained, pretrained2d=pretrained2d, simMIM_pretrained=simMIM_pretrained, window_size=window_size, drop_rate=drop_rate, num_classes=num_classes))
    return model

def swin_large_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[2, 2, 18, 2],                
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


## set recommended archs
# 3D swin
swin_small_3D = swin_small_patch4_window8_3D
swin_base_3D = swin_base_patch4_window8_3D
swin_large_3D = swin_large_patch4_window8_3D

