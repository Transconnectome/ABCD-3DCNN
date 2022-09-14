# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from model.layers.patch_embed import PatchEmbed_2D, PatchEmbed_3D
from .vision_transformer import  Block

from util.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, RelativePositionBias2D, RelativePositionBias3D


class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """

    
    def __init__(self, img_size=256, patch_size=16, in_channels=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., attn_drop=.0, drop=.0, drop_path=.0, norm_layer=nn.LayerNorm, norm_pix_loss=False, use_sincos_pos=False, spatial_dims=3, mask_ratio=0.75):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.mask_ratio = mask_ratio
        if self.spatial_dims == 2:
            self.patch_embed = PatchEmbed_2D(img_size, patch_size, self.in_channels, embed_dim)
        elif self.spatial_dims == 3:
            self.patch_embed = PatchEmbed_3D(img_size, patch_size, self.in_channels, embed_dim)
        self.use_sincos_pos = use_sincos_pos
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  #  using absolute positional bias for positional encoding. fixed sin-cos embedding  

        # attention block 
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=drop,  attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        
        # the last layer normalization
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # positional embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  #  using absolute positional bias for positional encoding. fixed sin-cos embedding  
        

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if self.spatial_dims == 2:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * self.in_channels, bias=True) # decoder to patch
        elif self.spatial_dims == 3: 
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * self.in_channels, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.use_sincos_pos:
            if self.spatial_dims == 2:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            elif self.spatial_dims == 3:
                pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1],int(round(self.patch_embed.num_patches**(1/3))), cls_token=True)
            
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
            if self.spatial_dims == 2:
                decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            elif self.spatial_dims == 3:
                decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(round(self.patch_embed.num_patches**(1/3))), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

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

    def unpatchify_2D(self, x):
        """
        x: (N, L, patch_size**2 * in_channel)
        imgs: (N, in_channel, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = d = int(x.shape[1]**.5)
        assert h * w  == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, w * p))
        return imgs

    def unpatchify_3D(self, x):
        """
        x: (N, L, patch_size**3 * in_channel)
        imgs: (N, in_channel, H, W, D)
        """
        p = self.patch_embed.patch_size[0]
        h = w = d = int(round(x.shape[1]**(1/3)))
        assert h * w * d == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, self.in_channels))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        """
        1. ids_store의 value는 ids_shuffle의 index. 
        2. mask의 index & value는 ids_shuffle의 index & value와 표현하는 바가 똑같다. (오름 차순으로 정렬 했을 때, len_keep과 같은 갯수의 숫자들. 즉, network로 들어가는 갯수)
        3. 1과 2에 의해서, mask에서 ids_store를 torch.gather()의 index로 사용하는 것 = ids_shuffle에서 ids_store를 torch.gather()의 index로 사용하는 것
        4. 3은 바꿔서 말하면,  
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # about torch.gather(): https://data-newbie.tistory.com/709

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    # using absolute positional encoding for positional embedding. 

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) 
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed          # using absolute positional encoding for positional embedding

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W, D]
        pred: [N, L, p*p*p*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.spatial_dims == 2:
            target = self.patchify_2D(imgs)
        elif self.spatial_dims == 3:
            target = self.patchify_3D(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # In case 2D, [N, L, p*p*in_channels], in case 3D [N, L, p*p*p*in_channels]
        
        #loss = self.forward_loss(imgs, pred, mask)
        #return loss, pred, mask
        
        if self.spatial_dims == 2:
            target = self.patchify_2D(imgs)
        elif self.spatial_dims == 3:
            target = self.patchify_3D(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return pred, target, mask 
        

        


def mae_vit_base_patch16_dec512d8b_2D(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,              # original embed_dim = 768
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b_2D(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,              # original embed_dim = 1024
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b_2D(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,              # original embed_dim = 1280
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=2, **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_3D(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,              # original encoder embed_dim = 768
        decoder_embed_dim=576, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b_3D(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,              # original encoder embed_dim = 1024
        decoder_embed_dim=576, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b_3D(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16,              # original encoder embed_dim = 1280
        decoder_embed_dim=576, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), spatial_dims=3, **kwargs)
    return model


## set recommended archs
# 2D ViT
mae_vit_base_patch16_2D = mae_vit_base_patch16_dec512d8b_2D  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_2D = mae_vit_large_patch16_dec512d8b_2D  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_2D = mae_vit_huge_patch14_dec512d8b_2D  # decoder: 512 dim, 8 blocks
# 3D ViT
mae_vit_base_patch16_3D = mae_vit_base_patch16_dec512d8b_3D  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_3D = mae_vit_large_patch16_dec512d8b_3D  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_3D = mae_vit_huge_patch14_dec512d8b_3D  # decoder: 512 dim, 8 blocks
