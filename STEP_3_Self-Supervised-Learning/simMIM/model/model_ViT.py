# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ViT: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# --------------------------------------------------------

from functools import partial

import numpy as np
from sklearn.feature_selection import SelectFdr

import torch
import torch.nn as nn

from model.layers.patch_embed import PatchEmbed_2D, PatchEmbed_3D
from .vision_transformer import  Block

from util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, resize_pos_embed
from .layers.helpers import to_3tuple
from util.utils import _n2p

class VisionTransformer3D(nn.Module):

    def __init__(self, 
                 pretrained=None,
                 pretrained2d=True,
                 simMIM_pretrained=False, 
                 img_size=256, 
                 patch_size=16, 
                 in_channels=1,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 mlp_ratio=4., 
                 attn_drop=.0, 
                 drop=.0, 
                 drop_path=.0, 
                 norm_layer=nn.LayerNorm, 
                 num_classes=1, 
                 global_pool='token', 
                 use_cls_tokens=True, 
                 fc_norm=None, 
                 use_rel_pos_bias=False, 
                 use_sincos_pos=True, 
                 spatial_dims=3, 
                 imagenet_pretrained_weight: str=None):
        """
        Args:
            pretrained (str): directory of pretrained model  
            pretrained2d (bool): whther the pretrained model trained with imagenet22k dataset or not 
            simMIM_pretrained (bool): whether the pretrained model is trained via simMIM or not
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            attn_drop (float): attention dropout rate
            drop (float): dropout rate
            drop_path (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer 
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            use_cls_tokens (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            use_rel_pos_bias (bool): use relative positional bias (applied to swin transformer)
            imagenet_pretrained_weight (str): 
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # ViT encoder specifics
        self.pretrained = pretrained 
        self.pretrained2d = pretrained2d
        self.simMIM_pretrained = simMIM_pretrained
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size = to_3tuple(patch_size)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.global_pool = global_pool
        self.spatial_dims = spatial_dims
        self.use_sincos_pos = use_sincos_pos
        self.use_rel_pos_bias = use_rel_pos_bias
        self.num_prefix_tokens = 1 if use_cls_tokens else 0
        global_pool == 'avg' if fc_norm is None else fc_norm

        if self.spatial_dims == 2:
            self.patch_embed = PatchEmbed_2D(img_size, patch_size, self.in_channels, embed_dim)
        elif self.spatial_dims == 3:
            self.patch_embed = PatchEmbed_3D(img_size, patch_size, self.in_channels, embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding
        if use_rel_pos_bias:    # using relative positional bias for positional encoding
            self.pos_embed = None  
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  #  using absolute positional bias for positional encoding. fixed sin-cos embedding  

        self.pos_drop = nn.Dropout(p=drop)

        # attention block 
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=drop,  attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None, spatial_dims=self.spatial_dims)
            for i in range(depth)])

        # the last layer normalization
        self.norm = norm_layer(embed_dim) if not self.global_pool else nn.Identity()
        
        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if self.global_pool else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # --------------------------------------------------------------------------
        self.initialize_weights(imagenet_pretrained_weight)

    def initialize_weights(self, imagenet_pretrained_weight=None):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.use_sincos_pos:
            if self.spatial_dims == 2:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            elif self.spatial_dims == 3:
                pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1],int(round(self.patch_embed.num_patches**(1/3))), cls_token=True)
            
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if self.num_classes > 0:
            torch.nn.init.normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(0.001)
            self.head.bias.data.mul_(0.001)

        if self.pretrained: 
            if isinstance(self.pretrained, str): 
                if self.pretrained2d: 
                    self.load_pretrained2d()    # load ImageNet22k pretrained model  
                else:
                    if self.simMIM_pretrained:
                        # load simMIM pretrained model 
                        self.load_simMIM_pretrained()
                    else:
                        # load finetuning model
                        state_dict = torch.load(self.pretrained, map_location='cpu')
                        self.load_state_dict(state_dict['model'])


    def load_pretrained2d(self):
        # load numpy file of imagenet pretrained model
        w = np.load(self.pretrained)
        # load weight on positional encoding only if absolute positional encoding is True 
        if self.use_sincos_pos:
            if w['Transformer/posembed_input/pos_embedding'].shape != self.pos_embed.shape:
                pos_embed_w = _n2p(w['Transformer/posembed_input/pos_embedding'], t=False)
                pos_embed_w = resize_pos_embed(     # resize pos embedding when different size from pretrained weights
                        pos_embed_w, 
                        self.pos_embed,
                        getattr(self,'num_prefix_tokens', 1),
                        self.patch_embed.grid_size
                )
            setattr(self, 'pos_embed.data', pos_embed_w)

        # load weight on attention blocks 
        for i, block in enumerate(self.blocks):
            # initial norm layer
            setattr(self, 'blocks[%s].norm1.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_0/scale' % str(i)]))
            setattr(self, 'blocks[%s].norm1.bias.data' % str(i),  _n2p(w['Transformer/encoderblock_%s/LayerNorm_0/bias' % str(i)]))
            # attention qkv parameters 
            setattr(self, 'blocks[%s].attn.qkv.weight.data' % str(i), torch.cat([_n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/%s/kernel' % (str(i), n)], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            setattr(self, 'blocks[%s].attn.qkv.bias.data' % str(i), torch.cat([_n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/%s/bias' % (str(i), n)], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            # attention projection layer
            setattr(self, 'blocks[%s].attn.proj.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/kernel' % str(i)]))
            setattr(self, 'blocks[%s].attn.proj.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MultiHeadDotProductAttention_1/out/bias' % str(i)]))
            # last norm layer
            setattr(self, 'blocks[%s].norm2.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_2/scale' % str(i)]))
            setattr(self, 'blocks[%s].norm2.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/LayerNorm_2/scale' % str(i)]))
            # fc layer 
            setattr(self, 'blocks[%s].mlp.fc1.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_0/kernel' % str(i)]))
            setattr(self, 'blocks[%s].mlp.fc1.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_0/bias' % str(i)]))
            setattr(self, 'blocks[%s].mlp.fc2.weight.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_1/kernel' % str(i)]))
            setattr(self, 'blocks[%s].mlp.fc2.bias.data' % str(i), _n2p(w['Transformer/encoderblock_%s/MlpBlock_3/Dense_1/bias' % str(i)]))                
        #load weight on the last norm layer
        setattr(self, 'norm.weight.data', _n2p(w['Transformer/encoder_norm/scale']))
        setattr(self, 'norm.bias.data', _n2p(w['Transformer/encoder_norm/bias']))
        
        del w
        torch.cuda.empty_cache()       
        print(f"=> loaded successfully '{self.pretrained}'")


    def load_simMIM_pretrained(self): 
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        prefix = 'encoder'

        # load weight on positional encoding only if absolute positional encoding is True 
        if self.use_sincos_pos:
            if state_dict['pos_embed'].shape != self.pos_embed.shape:
                pos_embed_w = state_dict['pos_embed']
                pos_embed_w = resize_pos_embed(     # resize pos embedding when different size from pretrained weights
                        pos_embed_w, 
                        self.pos_embed,
                        getattr(self,'num_prefix_tokens', 1),
                        self.patch_embed.grid_size
                )
            setattr(self, 'pos_embed.data', pos_embed_w)   


        # load patch embedding layers 
        setattr(self, 'patch_embed.proj.weight.data', state_dict['%s.patch_embed.proj.weight' % prefix])
        setattr(self, 'patch_embed.proj.bias.data', state_dict['%s.patch_embed.proj.bias' % prefix])

        # load weight on attention blocks 
        for i, block in enumerate(self.blocks):
            # initial norm layer
            setattr(self, 'blocks[%s].norm1.weight.data' % str(i), state_dict['%s.blocks.%s.norm1.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].norm1.bias.data' % str(i),  state_dict['%s.blocks.%s.norm1.bias' % (prefix, str(i))])
            # attention qkv parameters 
            setattr(self, 'blocks[%s].attn.qkv.weight.data' % str(i), state_dict['%s.blocks.%s.attn.qkv.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].attn.qkv.bias.data' % str(i), state_dict['%s.blocks.%s.attn.qkv.bias' % (prefix, str(i))])
            # attention projection layer
            setattr(self, 'blocks[%s].attn.proj.weight.data' % str(i), state_dict['%s.blocks.%s.attn.proj.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].attn.proj.bias.data' % str(i), state_dict['%s.blocks.%s.attn.proj.bias' % (prefix, str(i))])
            if self.use_rel_pos_bias:
                # attention relative position bias. Do not need to load relative_position_index  
                setattr(self, 'blocks[%s].attn.relative_position_bias_table.data' % str(i), state_dict['%s.blocks.%s.attn.relative_position_bias_table' % (prefix, str(i))])
            # last norm layer
            setattr(self, 'blocks[%s].norm2.weight.data' % str(i), state_dict['%s.blocks.%s.norm2.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].norm2.bias.data' % str(i), state_dict['%s.blocks.%s.norm2.bias' % (prefix, str(i))])
            # fc layer 
            setattr(self, 'blocks[%s].mlp.fc1.weight.data' % str(i), state_dict['%s.blocks.%s.mlp.fc1.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].mlp.fc1.bias.data' % str(i), state_dict['%s.blocks.%s.mlp.fc1.bias' % (prefix, str(i))])
            setattr(self, 'blocks[%s].mlp.fc2.weight.data' % str(i), state_dict['%s.blocks.%s.mlp.fc2.weight' % (prefix, str(i))])
            setattr(self, 'blocks[%s].mlp.fc2.bias.data' % str(i), state_dict['%s.blocks.%s.mlp.fc2.bias' % (prefix, str(i))])                
        #load weight on the last norm layer
        setattr(self, 'norm.weight.data', state_dict['%s.norm.weight' % prefix])
        setattr(self, 'norm.bias.data', state_dict['%s.norm.bias' % prefix])
        
        del checkpoint
        del state_dict
        torch.cuda.empty_cache()       
        
        print(f"=> loaded successfully '{self.pretrained}'")



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify_2D(self, imgs):
        """
        imgs: (N, in_channel, H, W)
        x: (N, L, patch_size**2 * in_channel)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w  = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p, ))
        x = torch.einsum('nchpwq->nhwpqc', x) # Changing the shape of Tensor. This line is same as transpose and permutation. 
        x = x.reshape(shape=(imgs.shape[0], h * w , p**2 * self.in_channels))
        return x

    def patchify_3D(self, imgs):
        """
        imgs: (N, in_channel, H, W, D)
        x: (N, L, patch_size**3 * in_channel)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x) # Changing the shape of Tensor. This line is same as transpose and permutation. 
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * self.in_channels))
        return x


    def forward_features(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if self.pos_embed is not None:           # using absolute positional encoding for positional embedding. 
            x = x + self.pos_embed[:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:                                    # else using relative positional bias for positional embedding (if using relative positional bias, self.pos_embed = None)
            cls_token = self.cls_token

        # append cls token
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        if self.global_pool:
            return x
        else:
            return self.norm(x)
        

    def forward_head(self, x):
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = x[:, 0]
            x = self.fc_norm(x)     # self.fc_norm = nn.Identity()
        #if self.global_pool:
        #    x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]  
        #x = self.fc_norm(x)
        x = self.head(x)

        return x 

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x         


def vit_base_patch16_dec512d8b_2D(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,              # original embed_dim = 768
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model


def vit_large_patch16_dec512d8b_2D(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,              # original embed_dim = 1024
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model


def vit_huge_patch14_dec512d8b_2D(**kwargs):
    model = VisionTransformer3D(
        embed_dim=1280, depth=32, num_heads=16,              # original embed_dim = 1280
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model

def vit_base_patch16_dec512d8b_3D(**kwargs):
    model = VisionTransformer3D(
        embed_dim=1152, depth=12, num_heads=12,              # original encoder embed_dim = 768
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model

def vit_large_patch16_dec512d8b_3D(**kwargs):
    model = VisionTransformer3D(
        embed_dim=1024, depth=24, num_heads=16,              # original encoder embed_dim = 1024
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model


def vit_huge_patch14_dec512d8b_3D(**kwargs):
    model = VisionTransformer3D(
        embed_dim=1280, depth=32, num_heads=16,              # original encoder embed_dim = 1280
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model


## set recommended archs
# 2D ViT
vit_base_patch16_2D = vit_base_patch16_dec512d8b_2D  
vit_large_patch16_2D = vit_large_patch16_dec512d8b_2D  
vit_huge_patch14_2D = vit_huge_patch14_dec512d8b_2D  
# 3D ViT
vit_base_patch16_3D = vit_base_patch16_dec512d8b_3D 
vit_large_patch16_3D = vit_large_patch16_dec512d8b_3D  
vit_huge_patch14_3D = vit_huge_patch14_dec512d8b_3D 
