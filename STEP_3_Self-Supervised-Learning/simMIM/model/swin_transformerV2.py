
# Code reference: https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
import math
import logging
from functools import partial, reduce, lru_cache
from operator import mul
from einops import rearrange
from collections import OrderedDict
from typing import List, Optional, Tuple, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from model.layers.mlp import Mlp
from timm.models.layers import DropPath, trunc_normal_
from .layers.helpers import to_3tuple

def _mul(a: int, b: int) -> int:
    return a * b

def _reduce(iterable: Tuple[int, int, int]) -> int:
    value = iterable[0]
    for element in iterable[1:]:
        value = _mul(value, element)
    return value

def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    #windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)     # original version 
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int, int], D: int, H:int, W:int):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0] ,H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size: torch.Tensor, window_size: torch.Tensor, shift_size: Optional[torch.Tensor]=None):
    ## additional line for torchscript 
    if isinstance(x_size, torch.Tensor): 
        list_x_size : List[int]= []
        for i,_ in enumerate(x_size):
            list_x_size.append(x_size[i].item())
        x_size : Tuple[int, int, int] = (list_x_size[0], list_x_size[1], list_x_size[2])
    
    if isinstance(window_size, torch.Tensor):
        use_window_size : List[int] = []
        for i,_ in enumerate(window_size):
            use_window_size.append(window_size[i].item())
    else: 
        use_window_size : List[int]  = list(window_size)
    
    if shift_size is not None:
        if isinstance(shift_size, torch.Tensor):
            use_shift_size : List[int] = []
            for i,_ in enumerate(shift_size):
                use_shift_size.append(shift_size[i].item())
        else: 
            use_shift_size : List[int] = list(shift_size)
    else: 
        use_shift_size : List[int] = []
    
    for i,_ in enumerate(x_size):
        if x_size[i] <= window_size[i]:     
            use_window_size[i] = int(x_size[i])
            if shift_size is not None:
                use_shift_size[i] = 0

    #if shift_size is None:
    #    return (use_window_size[0], use_window_size[1], use_window_size[2])
    #else:
    #    return (use_window_size[0], use_window_size[1], use_window_size[2]), (use_shift_size[0], use_shift_size[1], use_shift_size[2])
    return (use_window_size[0], use_window_size[1], use_window_size[2]), (use_shift_size[0], use_shift_size[1], use_shift_size[2])

    ######################################
     # original version
    #use_window_size = list(window_size)
    #if shift_size is not None:
    #    use_shift_size = list(shift_size)
    #for i, in range(len(x_size)):
    #    if x_size[i] <= window_size[i]:   
    #        use_window_size[i] = int(x_size[i])
    #        if shift_size is not None:
    #            use_shift_size[i] = 0

    #if shift_size is None:
    #    return tuple(use_window_size)
    #else:
    #    return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): Window size in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False,  attn_drop=0., proj_drop=0.,
                 pretrained_window_size=(0, 0, 0)):

        super().__init__()
        self.dim = dim
        self.window_size : Tuple[int,int,int] = window_size  # Wd, Wh, Ww
        self.pretrained_window_size : Tuple[int,int,int] = pretrained_window_size    # Wd, Wh, Ww
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))        
        
        # get relative_coords_table
        relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)  
        relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)        
        relative_coords_table = torch.stack(torch.meshgrid(relative_coords_d, relative_coords_h, relative_coords_w)).permute(1, 2, 3, 0).contiguous().unsqueeze(0)    # 1, 2*Wd-1, 2*Wh-1, 2*Ww-1, 3
        if pretrained_window_size[0] > 0: 
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor , mask :Optional[torch.Tensor]=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv_bias = None 
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        # cosine attention 
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)   
        relative_position_bias = relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            assert mask is None
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (tuple[int]): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(4,4,4), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 pretrained_window_size=(0,0,0)):
        super().__init__()
        self.dim = dim
        self.input_resolution : Tuple[int,int,int] = input_resolution
        self.num_heads = num_heads
        self.window_size : Tuple[int,int,int] = window_size
        self.shift_size : Tuple[int,int,int] = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint
        if min(self.input_resolution) <= min(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = (0,0,0) 
            self.window_size = self.input_resolution

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if (self.shift_size[0] > 0) and (self.shift_size[1] > 0) and (self.shift_size[2] > 0):
            # calculate attention mask for SW-MSA
            D, H, W = self.input_resolution
            img_mask = torch.zeros((1, D, H, W, 1)) 
            d_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0 
            for d in d_slices: 
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt+= 1 
            mask_windows = window_partition(img_mask, self.window_size) # nW, window_size, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else: 
            attn_mask = None 
        
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x): 
        D, H, W = self.input_resolution
        B, L, C = x.shape 
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x 
        x = x.view(B, D, H, W, C)

        # cyclic shift 
        if (self.shift_size[0] > 0) and (self.shift_size[1] > 0) and (self.shift_size[2] > 0): 
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1,2,3))
        else: 
            shifted_x = x 
        
        # partition windows 
        x_windows = window_partition(shifted_x, self.window_size)   # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)   # nW*B, window_size, window_size, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)    # nW*B, window_size, window_size, window_size, C

        # merge windows 
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)     # B D' H' W' C

        # reverse cyclic shift 
        if (self.shift_size[0] > 0) and (self.shift_size[1] > 0) and (self.shift_size[2] > 0):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size)[2], dims=(1,2,3)) 
        else: 
            x = shifted_x
        x = x.view(B, D * H * W, C)

        # Post Layer Norm
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN 
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x 



class PatchMerging3D(nn.Module):
    """ Patch Merging Layer
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution : Tuple[int,int,int] = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D*H*W, C).
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 ==0 and W % 2 ==0, f"x size ({D}*{H}*{W}) are not even."
        
        x = x.view(B, D, H, W, C)
        
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C   
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C  
        x = x.view(B, -1, 8 * C)    # B H/2*W/2 8*C

        x = self.reduction(x)
        x = self.norm(x)

        return x 


# cache each stage results
#@lru_cache()
@torch.jit.ignore
def compute_mask(D: int, H: int, W: int, window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int], device: torch.device):
    mask_size: Tuple[int, int, int, int, int] = (1, D, H, W, 1)
    img_mask = torch.zeros(mask_size, device=device, dtype=torch.float32, requires_grad=False)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Resolution of input feature.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (4,4,4).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (tuple[int]): Window size in pre-training.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pretrained_window_size=(0,0,0)):
        super().__init__()
        self.dim = dim 
        self.input_resolution: Tuple[int,int,int] = input_resolution
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] //22, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)])

        # patch merging layer 
        if downsample is not None: 
            self.downsample = downsample(input_resolution,dim=dim, norm_layer=norm_layer)
        else: 
            self.downsample = None 

    def forward(self, x: torch.Tensor): 
        for blk in self.blocks: 
            if self.use_checkpoint: 
                x = checkpoint.checkpoint(blk, x)
            else: 
                x = blk(x)
        if self.downsample is not None: 
            x = self.downsample(x)
        return x 

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        img_size (tuple[int]): Input image size. Default: (128,128,128).
        patch_size (tuple[int]): Patch token size. Default: (4,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=(128, 128, 128),patch_size=(4,4,4), in_channels=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]

        self.in_chans = in_channels
        self.embed_dim = embed_dim

        self.proj= nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor): 
        B, C, H, W, D = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1,2)  # B Pd*Ph*Pw C
        if self.norm is not None: 
            x = self.norm(x)
        return x

