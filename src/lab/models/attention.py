"""A tiny causal-attention Transformer — the control model.

Attention keeps an unbounded KV cache (capacity grows with sequence length), so
it should recall losslessly up to context length and be ~flat in K. It is the
sanity baseline: if the attention control is NOT flat/high, the MP0 cell is
mis-specified and the measurement is treated as invalid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # use the fused, memory-efficient SDPA with a causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AttentionLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2, n_heads: int = 4, max_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).clamp_max(self.max_len - 1)
        x = self.embed(tokens) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
