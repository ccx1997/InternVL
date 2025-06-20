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

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func, unpad_input, pad_input = None, None, None, None, None

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

        if self.fused_attn and FLASH_ATTN_AVAILABLE:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)

            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None:
                # rope needs (B, H, N, D) but q is (B, N, H, D), so we transpose
                q = self.rope(q.transpose(1, 2), pos).transpose(1, 2)
                k = self.rope(k.transpose(1, 2), pos).transpose(1, 2)

            qkv = torch.stack([q, k, v], dim=2)

            if mask is None:
                qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
                cu_seqlens = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=qkv.device)
                max_s = N
                x = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale, causal=False
                )
                x = x.reshape(B, N, C)
            else:
                qkv = qkv.reshape(B, N, 3 * C)
                x_unpad, indices, cu_seqlens, max_s = unpad_input(qkv, mask)
                x_unpad = x_unpad.reshape(-1, 3, self.num_heads, self.head_dim)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale, causal=False
                )
                output_unpad = output_unpad.reshape(-1, C)
                x = pad_input(output_unpad, indices, B, N)

        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            if self.fused_attn:  # SDPA path
                attn_mask = None
                if mask is not None:
                    # The expected mask shape for SDPA is (B, H, N, N) or broadcastable
                    attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, N, -1).contiguous()
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0)
            else:  # Naive path
                warnings.warn("Using naive attention. This is NOT RECOMMENDED.")
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                if mask is not None:
                    attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, N, N)
                    attn = attn.masked_fill(attn_mask.logical_not(), float('-inf'))
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
