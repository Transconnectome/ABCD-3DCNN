# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Video Swin: https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/mmaction/models/backbones/swin_transformer.py#L12 
# --------------------------------------------------------
from itertools import repeat
import collections.abc
from functools import partial
from typing import Optional, Tuple

import numpy as np

from timm.models.layers import trunc_normal_

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, trunc_normal_




# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_3tuple = _ntuple(3)


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
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(attn.device))).exp()
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
        x = self.proj(x).flatten(2).transpose(1,2) # B Pd*Ph*Pw C
        if self.norm is not None: 
            x = self.norm(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop= nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SwinTransformer3D_v2(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (tuple[int]): Input image size. Default: (128,128,128).
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (tuple(int)): Window size. Default: (8, 8, 8).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 subject_data,
                 args, 
                 pretrained=None,
                 pretrained2d=False,
                 simMIM_pretrained=False, 
                 img_size=(128,128,128),
                 patch_size=(4,4,4),
                 num_classes=1,
                 in_channels=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(8,8,8),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.5,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0,0,0,0]):
        super().__init__()
        self.subject_data = subject_data
        self.cat_target = args.cat_target
        self.num_target = args.num_target 
        self.target = args.cat_target + args.num_target

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.simMIM_pretrained = simMIM_pretrained

        self.img_size: Tuple[int,int,int] = img_size
        self.window_size: Tuple[int,int,int] = window_size 
        self.patch_size: Tuple[int,int,int] = patch_size
        self.pretrained_window_sizes = [to_3tuple(i) for i in pretrained_window_sizes]
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.frozen_stages = frozen_stages
        self.in_channels = in_channels
        self.norm_layer = norm_layer

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        
        self.pos_drop : nn.Module = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer),
                                  patches_resolution[2] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=self.pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        
        # the last norm layer
        self.norm = norm_layer(self.num_features)

        # the last pooling layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # prediㅊtion head
        self.head =  self._make_fclayers()

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


    def _make_fclayers(self):
        FClayer = []
        for cat_label in self.cat_target:
            self.out_dim = len(self.subject_data[cat_label].value_counts())
            FClayer.append(nn.Sequential(nn.Linear(self.num_features, self.out_dim)))
        for num_label in self.num_target:
            FClayer.append(nn.Sequential(nn.Linear(self.num_features, 1)))
        return nn.ModuleList(FClayer)  
        

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
    
    def forward_features(self, x: torch.Tensor): 
        x = self.patch_embed(x)     # B L C == B D*H*W C 
        
        x = self.pos_drop(x)

        for layer in self.layers: 
            x = layer(x)    # B L C == B D*H*W C 
        
        x = self.norm(x)    # B L C == B D*H*W C 
        x = self.avgpool(x.transpose(1,2))  # B C 1
        x = torch.flatten(x, 1)
        return x 

    def forward(self, x: torch.Tensor): 
        results = {} 
        x = self.forward_features(x)
        for i in range(len(self.head)):
            results[self.target[i]] = self.head[i](x)
        return results


def swinV2_tiny_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D_v2(
        patch_size=(4,4,4), window_size=(8,8,8), depths=[2, 2, 6, 2], embed_dim=96, num_heads=[3, 6, 12, 24],              
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.2, **kwargs)
    return model

def swinV2_small_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D_v2(
        patch_size=(4,4,4), window_size=(10,10,10), depths=[2, 2, 18, 2], embed_dim=96, num_heads=[3, 6, 12, 24],              
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.3, **kwargs)
    return model

def swinV2_base_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D_v2(
        patch_size=(10,10,10), window_size=(8,8,8), depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32],                 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.5, **kwargs)
    return model

def swinV2_large_patch4_window8_3D(**kwargs):
    model = SwinTransformer3D_v2(
        patch_size=(4,4,4), window_size=(8,8,8), depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48],                
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.2, **kwargs)
    return model



## set recommended archs
# 3D swin
swinV2_tiny_3D = swinV2_tiny_patch4_window8_3D
swinV2_small_3D = swinV2_small_patch4_window8_3D
swinV2_base_3D = swinV2_base_patch4_window8_3D
swinV2_large_3D = swinV2_large_patch4_window8_3D

