"""Tiny training loop for one MP0 cell -> recall accuracy.

Online synthetic data (an infinite stream of fresh MQAR batches), fixed seeds,
AMP on CUDA, early stop once validation recall saturates. Returns the per-item
recall accuracy `ā` for the cell — the single number MP0 needs.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Determinism: set the cuBLAS workspace BEFORE any CUDA op (module import time, before
# the first matmul) so run-to-run variance from nondeterministic GPU reductions is removed
# -- otherwise the same (seed, config) cell can land at very different optima.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from .models.attention import AttentionLM
from .models.diagonal_ssm import DiagonalSSM
from .tasks import mqar


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def block_of(name: str) -> str:
    """Partition every parameter into one named block (MP0b-8 ablation)."""
    for b in ("embed", "to_q", "to_k", "to_v", "to_a", "head"):
        if b in name:
            return b
    if name.startswith("norm."):       # final LayerNorm travels with the head
        return "head"
    return "out"                        # layer .out.* and layer .norm.*


def apply_hybrid(model, arch, vocab, mp, state_dim, hy) -> None:
    """Assemble a hybrid initialisation: one block from the donor, the rest from the
    base. direction 'W_block' = donor W into base L (does adding W's block rescue L?);
    'L_block' = donor L into base W (does removing W's block break it?)."""
    ws, ls = int(hy["winner_seed"]), int(hy["loser_seed"])
    block, direction = hy["block"], hy["direction"]
    torch.manual_seed(ws); W = build_model(arch, vocab, mp, state_dim).state_dict()
    torch.manual_seed(ls); L = build_model(arch, vocab, mp, state_dim).state_dict()
    base, donor = (L, W) if direction == "W_block" else (W, L)
    sd = {name: (donor[name] if block_of(name) == block else base[name]) for name in base}
    model.load_state_dict(sd)


def apply_interp(model, arch, vocab, mp, state_dim, hy) -> None:
    """MP0b-11: linear interpolation of inits, theta_a = (1-a) W + a L. Probes the
    landscape geometry on the straight line between a winner and a loser."""
    ws, ls, a = int(hy["W_seed"]), int(hy["L_seed"]), float(hy["alpha"])
    torch.manual_seed(ws); W = build_model(arch, vocab, mp, state_dim).state_dict()
    torch.manual_seed(ls); L = build_model(arch, vocab, mp, state_dim).state_dict()
    model.load_state_dict({n: (1.0 - a) * W[n] + a * L[n] for n in W})


def apply_perturb(model, arch, vocab, mp, state_dim, hy) -> None:
    """MP0b-11: theta = base + eps * noise, noise scaled per-parameter by its own std.
    Probes whether rescue depends on perturbation MAGNITUDE (geometry) vs direction."""
    base_seed, noise_seed, eps = int(hy["base_seed"]), int(hy["noise_seed"]), float(hy["eps"])
    torch.manual_seed(base_seed); base = build_model(arch, vocab, mp, state_dim).state_dict()
    g = torch.Generator().manual_seed(noise_seed)
    sd = {}
    for n, p in base.items():
        if p.is_floating_point() and p.dim() > 0 and p.numel() > 1:
            sd[n] = p + eps * (torch.randn(p.shape, generator=g) * p.std())
        else:
            sd[n] = p
    model.load_state_dict(sd)


def build_model(arch: str, vocab: int, mp: dict[str, Any], state_dim: int) -> torch.nn.Module:
    d_model = int(mp.get("d_model", 128))
    n_layers = int(mp.get("n_layers", 2))
    if arch == "diag_ssm":
        return DiagonalSSM(vocab, d_model, state_dim=state_dim, n_layers=n_layers,
                           chunk=int(mp.get("scan_chunk", 8)))
    if arch == "attention":
        return AttentionLM(
            vocab, d_model, n_layers=n_layers,
            n_heads=int(mp.get("n_heads", 4)), max_len=int(mp.get("seq_len", 1024)) + 8,
        )
    raise ValueError(f"unknown arch: {arch}")


@torch.no_grad()
def _recall(model: torch.nn.Module, batches: list[mqar.MQARBatch]) -> float:
    model.eval()
    correct = total = 0
    for batch in batches:
        logits = model(batch.tokens)
        pred = logits.argmax(dim=-1)
        mask = batch.targets != -100
        correct += int((pred[mask] == batch.targets[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / max(total, 1)


def train_cell(arch: str, state_dim: int, num_pairs: int, seed: int, mp: dict[str, Any]) -> dict[str, Any]:
    """Train one (arch, N, K) cell at a fixed seed.

    Returns {accuracy, steps, k_correct, max_grad_norm, final_loss, history},
    where history is a list of {step, loss, grad_norm, eval_acc} samples — the
    optimization diagnostics needed to tell capacity limits from training failure.
    Stabilisers: gradient clipping + a short linear LR warmup.
    """
    device = resolve_device(str(mp.get("device", "auto")))
    V = int(mp.get("vocab_size", 64))
    Q = int(mp.get("num_queries", 32))
    batch_size = int(mp.get("batch_size", 32))
    steps = int(mp.get("steps", 2000))
    # attention needs a lower lr at long sequences (else it stalls at high K)
    lr = float(mp.get("attention_lr", mp.get("lr")) if arch == "attention" else mp.get("lr", 3e-3))
    amp = bool(mp.get("amp", True)) and device.type == "cuda"
    early = float(mp.get("early_stop_acc", 0.99))
    grad_clip = float(mp.get("grad_clip", 1.0))
    warmup = int(mp.get("warmup_steps", max(50, steps // 20)))
    log_every = int(mp.get("log_every", 100))

    # cap this process's share of VRAM so several cells can co-reside on one GPU
    frac = mp.get("mem_fraction")
    if frac and device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(float(frac), device.index or 0)
        except Exception:
            pass

    # init_seed controls model initialisation; data_seed controls the data stream.
    # Default both to `seed` (coupled); decouple them to test init-basin vs data-path.
    init_seed = int(mp.get("init_seed", seed))
    data_seed = int(mp.get("data_seed", seed))
    torch.manual_seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:                               # force the deterministic (math) SDPA backend: flash / mem-efficient
        torch.backends.cuda.enable_flash_sdp(False)          # attention have NON-deterministic backward,
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # which would break A3 for the attention arch.
        torch.backends.cuda.enable_math_sdp(True)            # (no-op for diag_ssm; cheap at seq<=128).
    except Exception:
        pass
    gen = torch.Generator().manual_seed(data_seed + 1)
    eval_gen = torch.Generator().manual_seed(data_seed + 9973)
    # MP0b-16 critical-window probe (opt-in): switch the TRAINING data stream to a new
    # seed at step `data_switch_at`. Eval set stays fixed (tied to data_seed) so accuracy
    # is comparable across switch points. Absent params => identical to prior behaviour.
    switch_at = int(mp.get("data_switch_at", 0))
    switch_seed = mp.get("data_switch_seed", None)

    vocab = V + num_pairs
    model = build_model(arch, vocab, mp, state_dim).to(device)
    # MP0b-7 causal intervention: force the initial decay-gate bias (the predicted
    # winning-ticket coordinate). Negative -> a near the A_MIN floor (more forgetting).
    idb = mp.get("init_decay_bias")
    if idb is not None and hasattr(model, "layers"):
        with torch.no_grad():
            for layer in model.layers:
                if hasattr(layer, "to_a"):
                    layer.to_a.bias.fill_(float(idb))
    if mp.get("hybrid"):                                  # MP0b-8 causal block ablation
        apply_hybrid(model, arch, vocab, mp, state_dim, mp["hybrid"])
    if mp.get("interp"):                                  # MP0b-11 geometry: interpolation
        apply_interp(model, arch, vocab, mp, state_dim, mp["interp"])
    if mp.get("perturb_mag"):                             # MP0b-11 geometry: perturbation magnitude
        apply_perturb(model, arch, vocab, mp, state_dim, mp["perturb_mag"])
    weight_decay = float(mp.get("weight_decay", 0.01))   # grokking lever (MP0b-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def fresh(generator):
        return mqar.make_batch(batch_size, num_pairs, V, Q, generator, device=device)

    eval_batches = [fresh(eval_gen) for _ in range(2)]

    acc = 0.0
    done = 0
    max_gnorm = 0.0
    nonfinite = 0
    last_loss = float("nan")
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        # short linear LR warmup, then constant
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / max(1, warmup))
        model.train()
        if switch_seed is not None and step == switch_at + 1:   # MP0b-16: swap training stream
            gen = torch.Generator().manual_seed(int(switch_seed) + 1)
        batch = fresh(gen)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(batch.tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), batch.targets.view(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                   # unscale before clipping
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        scaler.step(opt)
        scaler.update()
        done = step
        last_loss = float(loss.detach())
        if math.isfinite(gnorm):
            max_gnorm = max(max_gnorm, gnorm)
        else:
            nonfinite += 1                                     # spike; AMP scaler skips the step
        if step % log_every == 0 or step == steps:
            acc = _recall(model, eval_batches)
            history.append({"step": step, "loss": round(last_loss, 4),
                            "grad_norm": round(gnorm, 4), "eval_acc": round(acc, 4)})
            if acc >= early:
                break

    acc = _recall(model, eval_batches)
    return {"accuracy": acc, "steps": done, "k_correct": acc * num_pairs,
            "max_grad_norm": round(max_gnorm, 4), "nonfinite_steps": nonfinite,
            "final_loss": round(last_loss, 4), "history": history}
