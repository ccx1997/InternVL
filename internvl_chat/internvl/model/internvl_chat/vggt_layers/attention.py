# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func, flash_attn_varlen_func, unpad_input, pad_input = None, None, None, None

# try:
#     from xformers.ops import memory_efficient_attention
#     XFORMERS_AVAILABLE = True
# except ImportError:
#     memory_efficient_attention = None
#     XFORMERS_AVAILABLE = False

XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use flash_attn or sdpa
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn and FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if mask is None:
                x = flash_attn_func(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, causal=False
                )
            else:
                q_unpad, indices, cu_q_lens, max_s = unpad_input(q, mask)
                k_unpad, _, cu_k_lens, _ = unpad_input(k, mask)
                v_unpad, _, _, _ = unpad_input(v, mask)
                x_unpad = flash_attn_varlen_func(
                    q_unpad, k_unpad, v_unpad, cu_q_lens, cu_k_lens, max_s, max_s,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    causal=False,
                )
                x = pad_input(x_unpad, indices, B, N)
            x = x.transpose(1, 2)
        elif self.fused_attn:
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(-2).expand(B, N, N).unsqueeze(1)
            x = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(B, self.num_heads, N, N)
                attn = attn.masked_fill(attn_mask.logical_not(), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, mask=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, mask=mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
