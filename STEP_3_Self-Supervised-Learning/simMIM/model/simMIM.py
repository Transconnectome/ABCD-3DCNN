# --------------------------------------------------------
# reference: https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial
from json import encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from .model_Swin import SwinTransformer3D
from .model_ViT import VisionTransformer3D


class SwinTransformerForSimMIM(SwinTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c') # B L C
        _, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        #if self.ape:
        #    x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, C, D, H, W)
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c') # B L C
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        #H = W = int(L ** 0.5)
        D = W = H = int(round(L**(1/3)))
        x = x.reshape(B, C, D, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        if self.pos_embed is not None:           # using absolute positional encoding for positional embedding. 
            x = x + self.pos_embed
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:                                    # else using relative positional bias for positional embedding (if using relative positional bias, self.pos_embed = None)
            cls_token = self.cls_token
        #cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x)

        #rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            #x = blk(x, rel_pos_bias=rel_pos_bias)
            x = blk(x)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        D = W = H = int(round(L**(1/3)))
        x = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 3 * 1, kernel_size=1),
            PixelShuffle3D(self.encoder_stride),
        )

        self.in_channels = self.encoder.in_channels
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_channels
        return loss, x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class PixelShuffle3D(nn.Module):
    '''
    reference: http://www.multisilicon.com/blog/a25332339.html
    '''
    def __init__(self,scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale 

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3 

        out_depth = in_depth * self.scale 
        out_height = in_height * self.scale 
        out_width = in_width * self.scale 

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous() 

        return output.view(batch_size, nOut, out_depth, out_height, out_width) 
    

def simMIM_swin_small_patch4_window8_3D(**kwargs):
    encoder = SwinTransformerForSimMIM(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=96, num_heads=[3, 6, 12, 24], window_size=(8,8,8),             
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=32)     # encoder_stride == prediction resolution. In the original paper no less than 1 /16 of original image size perform well
    return model

def simMIM_swin_base_patch4_window8_3D(**kwargs):
    encoder = SwinTransformerForSimMIM(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], window_size=(8,8,8),                
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=32)
    return model

def simMIM_swin_large_patch4_window8_3D(**kwargs):
    encoder = SwinTransformerForSimMIM(
        patch_size=4, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[2, 2, 18, 2], window_size=(8,8,8),                
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=32)
    return model

def simMIM_vit_base_patch16_dec512d8b_3D(**kwargs):
    encoder = VisionTransformerForSimMIM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,              # original encoder embed_dim = 768
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=16)
    return model

def simMIM_vit_large_patch16_dec512d8b_3D(**kwargs):
    encoder = VisionTransformerForSimMIM(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,              # original encoder embed_dim = 1024
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=16)
    return model

def simMIM_vit_huge_patch14_dec512d8b_3D(**kwargs):
    encoder = VisionTransformerForSimMIM(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,              # original encoder embed_dim = 1280
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    model = SimMIM(encoder=encoder, encoder_stride=16)
    return model

simMIM_swin_small_3D = simMIM_swin_small_patch4_window8_3D
simMIM_swin_base_3D = simMIM_swin_base_patch4_window8_3D
simMIM_swin_large_3D = simMIM_swin_large_patch4_window8_3D
simMIM_vit_base_patch16_3D = simMIM_vit_base_patch16_dec512d8b_3D
simMIM_vit_large_patch16_3D = simMIM_vit_large_patch16_dec512d8b_3D
simMIM_vit_huge_patch16_3D = simMIM_vit_huge_patch14_dec512d8b_3D
