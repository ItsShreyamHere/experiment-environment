"""A plain-PyTorch selective gated linear-attention recurrence — the pure
fixed-state model whose associative capacity scales with the state size N.

This is the model that lets MP0 actually *measure* b. The carried state is a
matrix S_t ∈ R^{N×d_v} updated by a gated outer product of a key (dim N) and a
value (dim d_v):

    S_t = diag(a_t) · S_{t-1} + k_t v_tᵀ        (selective decay a_t over N slots)
    o_t = q_tᵀ S_t                               (query reads the associative store)

The number of key→value bindings recoverable scales with the key dimension N, so
K*(N) grows with N and its slope gives b = slope·log V. The decay a_t and the
projections are input-dependent (selectivity, à la Mamba/GLA).

The recurrence is evaluated with a **chunk-parallel** scan (the standard GLA form):
within a chunk, contributions are dense matmuls; only T/chunk states are carried
sequentially. This is mathematically identical to the naive per-timestep loop
(`gated_linear_attention_sequential`, kept for the equivalence test) but turns a
T-deep autograd graph into a T/chunk-deep one — seconds instead of minutes per
cell, and tractable at the real grid sizes. No CUDA kernels / mamba-ssm / Triton.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gated_linear_attention_sequential(q, k, v, a):
    """Reference O(T) recurrence (used only to validate the chunked version)."""
    B, T, N = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, N, d_v)
    outs = []
    for t in range(T):
        S = a[:, t, :, None] * S + k[:, t, :, None] * v[:, t, None, :]
        outs.append((q[:, t, :, None] * S).sum(dim=1))
    return torch.stack(outs, dim=1)


def gated_linear_attention(q, k, v, a, chunk: int = 16):
    """Chunk-parallel gated linear attention. Identical maths to the sequential
    recurrence, but intra-chunk terms are matmuls and only T/chunk states recur.

    q, k : (B, T, N) non-negative features ; v : (B, T, d_v) ; a : (B, T, N) in (0,1).

    Always computed in fp32 with autocast disabled: the chunked form needs
    exp(±cumulative-log-decay), which overflows fp16 (≈6e4 at init) and would
    NaN the state. The sequential form is bounded step-by-step and doesn't need
    this; the chunked form does.
    """
    if torch.is_autocast_enabled() or q.dtype != torch.float32:
        with torch.autocast(device_type=q.device.type, enabled=False):
            return gated_linear_attention(q.float(), k.float(), v.float(), a.float(), chunk)
    B, T, N = q.shape
    d_v = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    if pad:
        # pad at the end with a=1 (no decay), k=v=q=0 (no contribution); outputs dropped
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
    Tp = T + pad
    nC = Tp // chunk

    qc = q.view(B, nC, chunk, N)
    kc = k.view(B, nC, chunk, N)
    vc = v.view(B, nC, chunk, d_v)
    log_a = torch.log(a.view(B, nC, chunk, N).clamp_min(1e-20))
    clog = torch.cumsum(log_a, dim=2)          # inclusive cumulative log-decay within chunk
    P = torch.exp(clog)                        # P_j = prod_{r<=j} a_r   (B,nC,chunk,N)
    P_last = P[:, :, -1, :]                    # decay across a full chunk (B,nC,N)

    q_tilde = qc * P                           # q_j ⊙ P_j
    k_tilde = kc * torch.exp(-clog)            # k_l / P_l
    # intra-chunk: lower-triangular (l<=j) scores, then @ V
    scores = torch.einsum("bcjn,bcln->bcjl", q_tilde, k_tilde)
    mask = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, 0.0)
    o_intra = torch.einsum("bcjl,bcld->bcjd", scores, vc)

    # inter-chunk: carry state across chunks (sequential over nC only)
    # state contribution of chunk c: kbar_l = (P_last / P_l) ⊙ k_l ; S_contrib = kbarᵀ V
    kbar = kc * torch.exp(clog[:, :, -1:, :] - clog)        # (B,nC,chunk,N)
    S_chunk = torch.einsum("bcln,bcld->bcnd", kbar, vc)     # per-chunk state delta (B,nC,N,d_v)

    outs = []
    S = q.new_zeros(B, N, d_v)
    for c in range(nC):
        o_inter = torch.einsum("bjn,bnd->bjd", q_tilde[:, c], S)   # read carried state
        outs.append(o_intra[:, c] + o_inter)
        S = P_last[:, c][:, :, None] * S + S_chunk[:, c]           # advance carried state
    O = torch.cat(outs, dim=1)                 # (B, Tp, d_v)
    return O[:, :T]


class GatedLinearAttentionLayer(nn.Module):
    # Decay floored at A_MIN to keep the chunked scan's exp(±cumulative-log-decay)
    # in a tiny, always-safe range. The intra-chunk term forms exp(-clog) on the
    # FULL (pre-mask) score matrix, so even a "finite but huge" exp (e.g. exp(31)≈4e13
    # at A_MIN=0.02) blows up gradients once multiplied by the unbounded elu+1
    # features -> inf. With chunk=8 and A_MIN=0.5: |clog| <= 8*0.69 ≈ 5.5 -> exp <= 245.
    # This only caps *forgetting* speed (<=50%/step), which a recall task does not need.
    A_MIN = 0.5

    def __init__(self, d_model: int, state_dim: int, chunk: int = 8):
        super().__init__()
        self.N = state_dim
        self.d_v = d_model
        self.chunk = chunk
        self.to_q = nn.Linear(d_model, state_dim)
        self.to_k = nn.Linear(d_model, state_dim)
        self.to_v = nn.Linear(d_model, d_model)
        self.to_a = nn.Linear(d_model, state_dim)   # input-dependent decay (selectivity)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        q = F.elu(self.to_q(u)) + 1.0
        k = F.elu(self.to_k(u)) + 1.0
        v = self.to_v(u)
        a = self.A_MIN + (1.0 - self.A_MIN) * torch.sigmoid(self.to_a(u))  # decay in [A_MIN, 1)
        O = gated_linear_attention(q, k, v, a, chunk=self.chunk)
        return u + self.out(self.norm(O))


class DiagonalSSM(nn.Module):
    """Embedding -> stacked selective gated-linear-attention layers -> LM head."""

    def __init__(self, vocab_size: int, d_model: int, state_dim: int, n_layers: int = 2,
                 chunk: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [GatedLinearAttentionLayer(d_model, state_dim, chunk=chunk) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))
