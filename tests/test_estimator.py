"""The estimator must recover a known slope and flag super-linear K*(N) as a kill."""

from __future__ import annotations

import math

from src.lab import estimator


def test_linear_recovers_b():
    log_V = 6.0
    # K* = 2*N exactly -> slope 2 -> b = 2*log_V
    per_n = {16: [32.0], 64: [128.0], 256: [512.0]}
    est = estimator.estimate(per_n, log_V, attention_acc=[0.99, 0.99])
    assert est["verdict"] == estimator.LINEAR
    assert math.isclose(est["b"], 2 * log_V, rel_tol=1e-6)
    assert estimator.verdict_to_outcome(est["verdict"]) == "survived"


def test_superlinear_is_a_kill():
    per_n = {16: [256.0], 64: [4096.0], 256: [65536.0]}  # K* = N^2 -> exponent 2
    est = estimator.estimate(per_n, 6.0, attention_acc=[0.99])
    assert est["verdict"] == estimator.SUPERLINEAR
    assert estimator.verdict_to_outcome(est["verdict"]) == "killed"


def test_invalid_when_control_not_flat():
    per_n = {16: [32.0], 64: [128.0]}
    est = estimator.estimate(per_n, 6.0, attention_acc=[0.2, 0.1])  # control failed
    assert est["verdict"] == estimator.INVALID
    assert est["b"] is None


def test_censored_grid_is_invalid():
    # K* maxes out at the smallest K for every N -> censored -> instrument refuses
    per_n = {16: [8.0], 32: [8.0], 64: [8.0]}
    grid = {(16, 8): 8.0, (16, 16): 2.5, (16, 32): 2.2,
            (32, 8): 8.0, (32, 16): 0.2, (32, 32): 3.3,
            (64, 8): 8.0, (64, 16): 0.2, (64, 32): 3.2}
    est = estimator.estimate(per_n, 6.0, attention_acc=[1.0], ssm_grid=grid)
    assert est["verdict"] == estimator.INVALID
    assert any("censored" in r for r in est["reasons"])


def test_monotonicity_axiom_drop_is_invalid():
    # Axiom A2: K_correct(K) must never drop. The pilot-3 pathology: 8 -> 16 -> 4.4.
    per_n = {16: [8.0], 32: [16.0], 64: [18.0]}
    grid = {(16, 8): 8.0, (16, 16): 8.1, (16, 32): 6.1,
            (32, 8): 8.0, (32, 16): 16.0, (32, 32): 4.4,   # <- drop: undertraining
            (64, 8): 8.0, (64, 16): 16.0, (64, 32): 18.3}
    est = estimator.estimate(per_n, 6.0, attention_acc=[1.0], ssm_grid=grid)
    assert est["verdict"] == estimator.INVALID
    assert any("A2" in r for r in est["reasons"])


def test_seed_disagreement_axiom_is_invalid():
    # Axiom A3: wildly different K* across seeds -> not trustworthy
    per_n = {16: [8.0, 8.0], 32: [16.0, 2.0], 64: [18.0, 18.0]}  # N=32 seeds disagree
    grid = {(16, 8): 8.0, (16, 16): 8.0, (32, 8): 8.0, (32, 16): 16.0,
            (64, 8): 8.0, (64, 16): 16.0, (64, 32): 18.0}
    est = estimator.estimate(per_n, 6.0, attention_acc=[1.0], ssm_grid=grid)
    assert est["verdict"] == estimator.INVALID
    assert any("A3" in r for r in est["reasons"])


def test_clean_linear_grid_still_valid():
    # monotone in N, K* achieved past the smallest K -> guards pass, slope read
    per_n = {16: [8.0], 32: [16.0], 64: [32.0]}
    grid = {(16, 8): 8.0, (16, 16): 8.0, (32, 8): 8.0, (32, 16): 16.0,
            (64, 8): 8.0, (64, 16): 16.0, (64, 32): 32.0}
    est = estimator.estimate(per_n, 6.0, attention_acc=[1.0], ssm_grid=grid)
    assert est["verdict"] == estimator.LINEAR


def test_bootstrap_ci_present_with_multiple_seeds():
    per_n = {16: [30.0, 34.0], 64: [120.0, 130.0], 256: [500.0, 520.0]}
    est = estimator.estimate(per_n, 6.0, attention_acc=[0.99])
    assert est["b_ci"][0] <= est["b"] <= est["b_ci"][1]
    assert est["repeatability"] is not None
