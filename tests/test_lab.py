"""A tiny model actually learns MQAR (beats chance) on CPU — the lab works."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.lab.tasks import mqar  # noqa: E402
from src.lab.train import train_cell  # noqa: E402
import torch  # noqa: E402


def test_mqar_batch_shapes_and_mask():
    g = torch.Generator().manual_seed(0)
    batch = mqar.make_batch(batch_size=4, num_pairs=4, vocab_size=16, num_queries=4, generator=g)
    assert batch.tokens.shape == batch.targets.shape
    # exactly num_queries scored positions per example (<= K)
    scored = (batch.targets != -100).sum(dim=1)
    assert int(scored.min()) == 4 and int(scored.max()) == 4
    assert batch.vocab_size == 16 + 4


def test_diag_ssm_beats_chance():
    mp = {
        "vocab_size": 16, "num_queries": 4, "d_model": 64, "n_layers": 2,
        "batch_size": 16, "steps": 500, "lr": 3e-3, "amp": False,
        "early_stop_acc": 0.95, "device": "cpu", "seq_len": 64,
    }
    out = train_cell("diag_ssm", state_dim=64, num_pairs=4, seed=0, mp=mp)
    # chance is 1/V = 1/16 ≈ 0.06; a working recurrence should clear several× that
    assert out["accuracy"] > 0.15
    assert out["k_correct"] == pytest.approx(out["accuracy"] * 4)
