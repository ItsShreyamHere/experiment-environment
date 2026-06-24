"""Multi-Query Associative Recall (MQAR) — the recall-capacity probe.

A batch is built deterministically from a torch.Generator seed. Each example:
  1. Context phase: K distinct keys, each bound to a random value (alphabet V),
     laid out as [key, value, key, value, ...]  (length 2K).
  2. Query phase: a random sample of `num_queries` stored keys is re-presented;
     the model must predict each key's value at the next position.

Loss/accuracy are computed ONLY at the query-key positions. `k_correct = acc * K`
estimates the number of items recalled (acc is the per-item recall probability,
estimated from the sampled queries), and `K* = max_K k_correct`.

Token layout in one shared vocabulary of size (V + K):
  ids [0, V)        -> value tokens
  ids [V, V + K)    -> key tokens
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MQARBatch:
    tokens: torch.Tensor   # (B, L) long
    targets: torch.Tensor  # (B, L) long, -100 where not scored
    vocab_size: int
    seq_len: int


def make_batch(
    batch_size: int,
    num_pairs: int,
    vocab_size: int,
    num_queries: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> MQARBatch:
    K = num_pairs
    V = vocab_size
    Q = min(num_queries, K)
    L = 2 * K + 2 * Q

    g = generator
    B = batch_size
    key_base = V  # key tokens start after value tokens

    tokens = torch.zeros(B, L, dtype=torch.long)
    targets = torch.full((B, L), -100, dtype=torch.long)
    qpos = 2 * K + 2 * torch.arange(Q)              # query-key positions (scored)

    # Per-example RNG draws are kept in the SAME order as the reference loop, so output
    # is byte-identical; only the inner per-position assignment is vectorised away.
    for b in range(B):
        values = torch.randint(0, V, (K,), generator=g)        # key -> value binding
        ctx_order = torch.randperm(K, generator=g)             # context order
        q_order = torch.randperm(K, generator=g)[:Q]           # queried keys (sample)
        # context: interleave [key, value, key, value, ...]
        ctx = torch.stack([key_base + ctx_order, values[ctx_order]], dim=1).reshape(-1)  # (2K,)
        qseq = torch.stack([key_base + q_order, values[q_order]], dim=1).reshape(-1)      # (2Q,)
        tokens[b] = torch.cat([ctx, qseq])
        targets[b, qpos] = values[q_order]                     # score the value at each query key

    vocab = V + K
    return MQARBatch(
        tokens=tokens.to(device),
        targets=targets.to(device),
        vocab_size=vocab,
        seq_len=L,
    )
