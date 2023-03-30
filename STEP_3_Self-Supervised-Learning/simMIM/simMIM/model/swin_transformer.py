
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
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, _reduce(window_size), C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int, int], B: int, D: int, H:int, W:int):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
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
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor , mask :Optional[torch.Tensor]=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
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
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(4,4,4), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size : Tuple[int,int,int] = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        #window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)    # original version 
        window_size, shift_size = get_window_size(torch.tensor((D,H,W)), torch.tensor(self.window_size), torch.tensor(self.shift_size))     # torchscript version

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = int((window_size[0] - D % window_size[0]) % window_size[0])
        pad_b = int((window_size[1] - H % window_size[1]) % window_size[1])
        pad_r = int((window_size[2] - W % window_size[2]) % window_size[2])
        
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        count = 0
        for i in shift_size: 
            if i > 0:
                count += 1 
        #if any(i > 0 for i in shift_size):
        if count > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        reverse_count = 0
        for i in shift_size: 
            if i > 0:
                reverse_count += 1
        #if any(i > 0 for i in shift_size):
        if reverse_count > 0:
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        """
        For torchscript, checkpoint function couldn't be used.
        """

        shortcut = x
#        if self.use_checkpoint:
#            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
#        else:
#            x = self.forward_part1(x, mask_matrix)
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

#        if self.use_checkpoint:
#            x = x + checkpoint.checkpoint(self.forward_part2, x)
#        else:
#            x = x + self.forward_part2(x)
        x = x + self.forward_part2(x)

        return x


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, D % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C   
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C     
        x = self.norm(x)
        x = self.reduction(x)

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
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        #window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)      # original version 
        window_size, shift_size = get_window_size(torch.tensor((D,H,W)), torch.tensor(self.window_size), torch.tensor(self.shift_size))     # torchscript version

        #x = rearrange(x, 'b c d h w -> b d h w c')     # original version
        #Dp = int(np.ceil(D / window_size[0])) * window_size[0]     # original version
        #Hp = int(np.ceil(H / window_size[1])) * window_size[1]     # original version
        #Wp = int(np.ceil(W / window_size[2])) * window_size[2]     # original version
        x = torch.einsum('bcdhw->bdhwc',x)      # torchscript version 
        Dp: int = int(torch.ceil(torch.tensor(D / window_size[0]))) * window_size[0]     # torchscript version
        Hp: int = int(torch.ceil(torch.tensor(H / window_size[1]))) * window_size[1]     # torchscript version
        Wp: int = int(torch.ceil(torch.tensor(W / window_size[2]))) * window_size[2]     # torchscript version
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        #x = rearrange(x, 'b d h w c -> b c d h w')     # original version
        x = torch.einsum('bdhwc->bcdhw', x)     # torchscript version
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        #self.patch_size = to_3tuple(patch_size)
        self.patch_size : int = patch_size

        self.in_chans = in_channels
        self.embed_dim = embed_dim

        self.proj : nn.Module = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm : nn.Module = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
        if D % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size - D % self.patch_size))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wd, Wh, Ww)

        return x