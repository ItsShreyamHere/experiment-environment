"""The chunk-parallel scan must equal the naive sequential recurrence."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from src.lab.models.diagonal_ssm import (  # noqa: E402
    gated_linear_attention,
    gated_linear_attention_sequential,
)


@pytest.mark.parametrize("T,chunk", [(16, 16), (40, 16), (64, 16), (37, 8)])
def test_chunked_matches_sequential(T, chunk):
    torch.manual_seed(0)
    B, N, d_v = 2, 24, 16
    q = torch.rand(B, T, N) + 0.1
    k = torch.rand(B, T, N) + 0.1
    v = torch.randn(B, T, d_v)
    a = torch.sigmoid(torch.randn(B, T, N))
    ref = gated_linear_attention_sequential(q, k, v, a)
    fast = gated_linear_attention(q, k, v, a, chunk=chunk)
    assert torch.allclose(ref, fast, atol=1e-4, rtol=1e-4)


def test_gradients_flow():
    torch.manual_seed(0)
    B, T, N, d_v = 2, 48, 16, 16
    q = (torch.rand(B, T, N) + 0.1).requires_grad_()
    k = (torch.rand(B, T, N) + 0.1).requires_grad_()
    v = torch.randn(B, T, d_v, requires_grad=True)
    a = torch.sigmoid(torch.randn(B, T, N)).requires_grad_()
    gated_linear_attention(q, k, v, a, chunk=16).sum().backward()
    assert q.grad is not None and v.grad is not None and a.grad is not None
