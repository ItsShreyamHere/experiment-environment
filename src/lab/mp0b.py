"""MP0b — Bimodal Convergence Investigation.

Not a measurement program and NOT a fix. ICP0 found that the same architecture +
the same hyperparameters + a different seed produces two outcomes for an overloaded
recall cell: a PLATEAU seed (reaches true K*) vs a STUCK-LOW seed (trapped ~half).
A3 correctly flags this as irreproducible. Here we *study* it before changing anything.

Experiment MP0b-1: run one fixed cell across many seeds, capture full training
trajectories, and characterise:
  - the distribution of final K_correct (truly bimodal? what fraction stuck?);
  - metastability (do stuck seeds stay flat, or jump late like grokking?);
  - the bifurcation step (when do plateau vs stuck trajectories separate?).

Candidate explanations to discriminate (later experiments): grokking, lottery-ticket,
metastability, first-order phase transition, critical LR, initialisation basin.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from ..db.connection import Database, utcnow
from ..phenomena import phenomenon as ph
from ..utils import logging as log
from ..utils.hashing import short_hash
from .icp0 import _train_worker

STAGE = "mp0b"
MP_ID = "MP0b"


def init_metrics(arch: str, vocab: int, mp: dict, state_dim: int, init_seed: int) -> dict[str, float]:
    """Properties of the freshly-initialised model (no training) — candidate predictors
    of the eventual basin. Computed deterministically from init_seed.
    The GatedLinearAttentionLayer projections are to_q/to_k (d_model->N), to_v, to_a
    (decay), and `out` (readout); there is no separate C matrix."""
    import torch
    from .train import build_model
    torch.manual_seed(init_seed)
    model = build_model(arch, vocab, mp, state_dim)
    m = {"param_norm": float(sum(p.detach().norm().item() ** 2 for p in model.parameters()) ** 0.5)}

    def srank(W):
        s = torch.linalg.svdvals(W.detach())
        sn = float(s[0])
        return sn, (float((s ** 2).sum()) / (sn ** 2) if sn > 0 else 0.0)

    if hasattr(model, "layers") and len(model.layers) and hasattr(model.layers[0], "to_q"):
        qsn, qsr, osn, abias = [], [], [], []
        for layer in model.layers:
            sn, sr = srank(layer.to_q.weight); qsn.append(sn); qsr.append(sr)
            osn.append(srank(layer.out.weight)[0])
            abias.append(float(layer.to_a.bias.detach().mean()))
        m["q_specnorm"] = sum(qsn) / len(qsn)
        m["q_stable_rank"] = sum(qsr) / len(qsr)
        m["out_specnorm"] = sum(osn) / len(osn)
        m["a_bias_mean"] = sum(abias) / len(abias)
    return m


def run_init_predictor(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-6: can the eventual basin be predicted from the INIT weights alone?
    For each init seed, compute init-weight metrics + train (fixed data) to its outcome;
    correlate each metric with final K_correct."""
    ip = mp["init_predictor"]
    n_inits = int(ip.get("n_inits", 16))
    data_seed = int(ip.get("data_seed", 0))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    V = int(mp.get("vocab_size", 16))
    vocab = V + K
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b" / "init_predictor"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] INIT-PREDICTOR: {n_inits} inits on {arch} N={N} K={K} (data_seed={data_seed}).")

    def unit(i):
        return f"{arch}:N{N}:K{K}:ip{i}"

    def args(i):
        m = dict(mp); m["init_seed"] = i; m["data_seed"] = data_seed; m.pop("init_predictor", None)
        return (arch, N, K, 0, m)

    done = db.done_units(STAGE)
    pending = [i for i in range(n_inits) if unit(i) not in done]

    def record(i, out):
        try:
            met = init_metrics(arch, vocab, mp, N, i)
        except Exception as exc:                              # never discard a trained result
            log.warn(f"[MP0b] init {i} metrics failed ({exc}); saving final_k only")
            met = {}
        (runs_dir / f"i{i}.json").write_text(json.dumps(
            {"init_seed": i, "final_k": out["k_correct"], **met}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(i), "done")
        log.dim(f"[MP0b] init {i}: final_k={out['k_correct']:.1f} metrics={ {k: round(v,2) for k,v in met.items()} }")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, args(i)): i for i in pending}
            for i in pending:
                db.mark(STAGE, unit(i), "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    i = futs[fut]
                    try:
                        _, out = fut.result()
                        record(i, out)
                    except Exception as exc:
                        db.mark(STAGE, unit(i), "failed", str(exc))
                        log.warn(f"[MP0b] init {i} failed: {exc}")
    else:
        for i in pending:
            _, out = _train_worker(args(i))
            record(i, out)

    # correlate each metric with final_k
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in runs_dir.glob("i*.json")]
    finals = [r["final_k"] for r in recs]
    metric_keys = [k for k in recs[0] if k not in ("init_seed", "final_k")]
    out: dict[str, Any] = {"n": len(recs), "finals": sorted(round(f, 1) for f in finals)}
    for mk in metric_keys:
        xs = [r[mk] for r in recs]
        out[f"corr_{mk}"] = round(_pearson(xs, finals), 3)
    log.success(f"[MP0b] init-predictor n={out['n']} correlations(final_k vs init metric): "
                + ", ".join(f"{mk}={out[f'corr_{mk}']}" for mk in metric_keys))
    return out


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def run_hybrid(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-8: causal block ablation. Swap one parameter block at a time between a
    robust winner (W) and loser (L); which block transfers the outcome?"""
    hy = mp["hybrid"]
    ws, ls = int(hy["winner_seed"]), int(hy["loser_seed"])
    blocks = list(hy.get("blocks", ["embed", "to_q", "to_k", "to_v", "to_a", "out", "head"]))
    directions = list(hy.get("directions", ["W_block", "L_block"]))
    data_seeds = list(hy.get("data_seeds", [0, 1, 2]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b" / "hybrid"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] HYBRID ablation W={ws} L={ls}: {blocks} x {directions} x {len(data_seeds)} seeds.")

    # build the cell list: hybrids + pure-W / pure-L controls
    cells = []  # (tag, mp-overrides)
    for s in data_seeds:
        cells.append((f"ctrl:W:s{s}", {"init_seed": ws, "data_seed": s}))
        cells.append((f"ctrl:L:s{s}", {"init_seed": ls, "data_seed": s}))
    for d in directions:
        for b in blocks:
            for s in data_seeds:
                cells.append((f"hyb:{d}:{b}:s{s}",
                              {"data_seed": s,
                               "hybrid": {"winner_seed": ws, "loser_seed": ls, "block": b, "direction": d}}))

    def unit(tag):
        return f"{arch}:N{N}:K{K}:{tag}"

    def args(over):
        m = {k: v for k, v in mp.items() if k != "hybrid"}
        m.update(over)
        return (arch, N, K, 0, m)

    done = db.done_units(STAGE)
    pending = [(tag, over) for tag, over in cells if unit(tag) not in done]

    def record(tag, out):
        (runs_dir / f"{tag.replace(':', '_')}.json").write_text(json.dumps(
            {"tag": tag, "final_k": out["k_correct"], "accuracy": out["accuracy"]}, indent=1),
            encoding="utf-8")
        db.mark(STAGE, unit(tag), "done")
        log.dim(f"[MP0b] {tag}: final_k={out['k_correct']:.1f}")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, args(over)): tag for tag, over in pending}
            for tag, _ in pending:
                db.mark(STAGE, unit(tag), "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    tag = futs[fut]
                    try:
                        _, out = fut.result()
                        record(tag, out)
                    except Exception as exc:
                        db.mark(STAGE, unit(tag), "failed", str(exc))
                        log.warn(f"[MP0b] {tag} failed: {exc}")
    else:
        for tag, over in pending:
            _, out = _train_worker(args(over))
            record(tag, out)

    return _analyze_hybrid(runs_dir, blocks, directions)


def _analyze_hybrid(runs_dir, blocks, directions) -> dict[str, Any]:
    import statistics
    recs = {}
    for p in runs_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        recs.setdefault(d["tag"].rsplit(":s", 1)[0], []).append(d["final_k"])
    def mean(key):
        vs = recs.get(key, [])
        return round(statistics.mean(vs), 1) if vs else None
    w, l = mean("ctrl:W"), mean("ctrl:L")
    out = {"control_W": w, "control_L": l, "W_block_into_L": {}, "L_block_into_W": {}}
    log.success(f"[MP0b] controls: pure-W={w}  pure-L={l}  (W wins, L stuck)")
    for d, label in [("W_block", "W_block_into_L"), ("L_block", "L_block_into_W")]:
        for b in blocks:
            m = mean(f"hyb:{d}:{b}")
            out[label][b] = m
        log.success(f"[MP0b] {label} (base={'L' if d=='W_block' else 'W'}): "
                    + "  ".join(f"{b}={out[label][b]}" for b in blocks))
    # flag transfers: W_block that flips L->win, or L_block that flips W->lose
    mid = ((w or 0) + (l or 0)) / 2
    out["W_block_transfers_win"] = [b for b in blocks if (out["W_block_into_L"][b] or 0) > mid]
    out["L_block_breaks_win"] = [b for b in blocks if (out["L_block_into_W"][b] or 99) < mid]
    log.info(f"[MP0b] blocks where W's block rescues L: {out['W_block_transfers_win']}")
    log.info(f"[MP0b] blocks where L's block breaks W: {out['L_block_breaks_win']}")
    return out


def run_perturb_control(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-9: does injecting a RANDOM block (a fresh seed, neither W nor L) into the
    loser rescue it as well as the winner's block does? If yes, "winning tickets
    transfer" dies — the rescue is mere perturbation / metastable barrier-kicking."""
    pc = mp["perturb_control"]
    L = int(pc.get("loser_seed", 1))
    blocks = list(pc.get("blocks", ["to_q", "to_a", "embed"]))
    donors = list(pc.get("donor_seeds", [0, 50, 51, 52]))   # 0 = real winner; rest random
    data_seeds = list(pc.get("data_seeds", [0, 1]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b" / "perturb"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] PERTURB CONTROL: donors {donors} (0=W, rest random) -> L={L}, blocks {blocks}.")

    cells = [(b, dn, s) for b in blocks for dn in donors for s in data_seeds]

    def unit(b, dn, s):
        return f"{arch}:N{N}:K{K}:pc:{b}:dn{dn}:s{s}"

    def args(b, dn, s):
        m = {k: v for k, v in mp.items() if k not in ("perturb_control", "hybrid")}
        m["data_seed"] = s
        m["hybrid"] = {"winner_seed": dn, "loser_seed": L, "block": b, "direction": "W_block"}
        return (arch, N, K, 0, m)

    done = db.done_units(STAGE)
    pending = [(b, dn, s) for b, dn, s in cells if unit(b, dn, s) not in done]

    def record(b, dn, s, out):
        (runs_dir / f"{b}_dn{dn}_s{s}.json").write_text(json.dumps(
            {"block": b, "donor": dn, "seed": s, "final_k": out["k_correct"]}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(b, dn, s), "done")
        log.dim(f"[MP0b] pc {b} donor={dn} s{s}: final_k={out['k_correct']:.1f}")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, args(b, dn, s)): (b, dn, s) for b, dn, s in pending}
            for b, dn, s in pending:
                db.mark(STAGE, unit(b, dn, s), "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    b, dn, s = futs[fut]
                    try:
                        _, out = fut.result()
                        record(b, dn, s, out)
                    except Exception as exc:
                        db.mark(STAGE, unit(b, dn, s), "failed", str(exc))
                        log.warn(f"[MP0b] pc {b} dn{dn} s{s} failed: {exc}")
    else:
        for b, dn, s in pending:
            _, out = _train_worker(args(b, dn, s))
            record(b, dn, s, out)

    # analyze: per block, mean final_k for real-W donor vs random donors
    out: dict[str, Any] = {"blocks": {}}
    for b in blocks:
        per_donor = {}
        for dn in donors:
            vs = [json.loads(p.read_text())["final_k"]
                  for p in runs_dir.glob(f"{b}_dn{dn}_s*.json")]
            per_donor[dn] = round(sum(vs) / len(vs), 1) if vs else None
        w = per_donor.get(donors[0])
        rand = [per_donor[d] for d in donors[1:] if per_donor[d] is not None]
        rand_mean = round(sum(rand) / len(rand), 1) if rand else None
        out["blocks"][b] = {"W_donor": w, "random_donors": [per_donor[d] for d in donors[1:]],
                            "random_mean": rand_mean}
        log.success(f"[MP0b] block {b}: W-donor rescues to {w}; random donors -> "
                    f"{out['blocks'][b]['random_donors']} (mean {rand_mean})")
    log.info("[MP0b] If random ≈ W => rescue is PERTURBATION, not a transferred ticket.")
    return out


def _jump(out) -> int | None:
    return next((h["step"] for h in (out.get("history") or []) if h.get("eval_acc", 0) >= 0.6), None)


def _pool_run(db, pending, mkargs, record, workers):
    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, mkargs(c)): c for c in pending}
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    c = futs[fut]
                    try:
                        _, out = fut.result(); record(c, out)
                    except Exception as exc:
                        log.warn(f"[MP0b] {c} failed: {exc}")
    else:
        for c in pending:
            _, out = _train_worker(mkargs(c)); record(c, out)


def run_interp(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-11a: linear interpolation W->L. Is there a sharp phase boundary in alpha?
    Does jump time diverge near it? Smooth or discontinuous landscape?"""
    ip = mp["interp"]
    ws, ls = int(ip.get("W_seed", 0)), int(ip.get("L_seed", 1))
    alphas = list(ip.get("alphas", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    seeds = list(ip.get("data_seeds", [0, 1, 2]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    runs_dir = cfg.data_path("runs") / "mp0b" / "interp"; runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] INTERP W={ws}->L={ls}: alphas={alphas} x {len(seeds)} seeds.")
    cells = [(a, s) for a in alphas for s in seeds]
    done = db.done_units(STAGE)

    def unit(a, s): return f"{arch}:N{N}:K{K}:interp:a{a}:s{s}"
    def mkargs(c):
        a, s = c
        m = {k: v for k, v in mp.items() if k != "interp"}
        m["data_seed"] = s; m["interp"] = {"W_seed": ws, "L_seed": ls, "alpha": a}
        return (arch, N, K, 0, m)
    def record(c, out):
        a, s = c
        (runs_dir / f"a{a}_s{s}.json").write_text(json.dumps(
            {"alpha": a, "seed": s, "final_k": out["k_correct"], "jump": _jump(out)}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(a, s), "done")
        log.dim(f"[MP0b] interp a={a} s{s}: final_k={out['k_correct']:.1f} jump={_jump(out)}")

    pending = [c for c in cells if unit(*c) not in done]
    for c in pending:
        db.mark(STAGE, unit(*c), "running")
    _pool_run(db, pending, mkargs, record, int(mp.get("workers", 3)))

    out = {"alphas": []}
    for a in alphas:
        recs = [json.loads(p.read_text()) for p in runs_dir.glob(f"a{a}_s*.json")]
        if not recs: continue
        fk = [r["final_k"] for r in recs]; jp = [r["jump"] for r in recs if r["jump"]]
        out["alphas"].append({"alpha": a, "mean_final": round(sum(fk)/len(fk), 1),
                              "finals": [round(x, 1) for x in fk],
                              "median_jump": int(sorted(jp)[len(jp)//2]) if jp else None})
        log.success(f"[MP0b] alpha={a}: mean_final={out['alphas'][-1]['mean_final']} "
                    f"finals={out['alphas'][-1]['finals']} median_jump={out['alphas'][-1]['median_jump']}")
    return out


def run_perturb_mag(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-11b: theta = L + eps*noise. Does rescue probability depend mainly on the
    perturbation MAGNITUDE eps (geometry) rather than the specific noise direction?"""
    pm = mp["perturb_mag"]
    base = int(pm.get("base_seed", 1))
    eps_values = list(pm.get("eps_values", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]))
    noise_seeds = list(pm.get("noise_seeds", [80, 81, 82, 83, 84, 85]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    runs_dir = cfg.data_path("runs") / "mp0b" / "perturb_mag"; runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] PERTURB-MAG base=L{base}: eps={eps_values} x {len(noise_seeds)} noise dirs.")
    cells = [(e, ns) for e in eps_values for ns in noise_seeds]
    done = db.done_units(STAGE)

    def unit(e, ns): return f"{arch}:N{N}:K{K}:pmag:e{e}:n{ns}"
    def mkargs(c):
        e, ns = c
        m = {k: v for k, v in mp.items() if k != "perturb_mag"}
        m["data_seed"] = 0; m["perturb_mag"] = {"base_seed": base, "noise_seed": ns, "eps": e}
        return (arch, N, K, 0, m)
    def record(c, out):
        e, ns = c
        (runs_dir / f"e{e}_n{ns}.json").write_text(json.dumps(
            {"eps": e, "noise": ns, "final_k": out["k_correct"]}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(e, ns), "done")
        log.dim(f"[MP0b] pmag eps={e} n{ns}: final_k={out['k_correct']:.1f}")

    pending = [c for c in cells if unit(*c) not in done]
    for c in pending:
        db.mark(STAGE, unit(*c), "running")
    _pool_run(db, pending, mkargs, record, int(mp.get("workers", 3)))

    out = {"eps": []}
    for e in eps_values:
        fk = [json.loads(p.read_text())["final_k"] for p in runs_dir.glob(f"e{e}_n*.json")]
        if not fk: continue
        rescued = sum(1 for x in fk if x > 9)
        out["eps"].append({"eps": e, "frac_rescued": round(rescued/len(fk), 2),
                           "mean_final": round(sum(fk)/len(fk), 1), "finals": [round(x, 1) for x in sorted(fk)]})
        log.success(f"[MP0b] eps={e}: rescued {rescued}/{len(fk)} (frac {out['eps'][-1]['frac_rescued']}) "
                    f"mean={out['eps'][-1]['mean_final']}")
    return out


def run_scale(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-12: the DEFLATIONARY ATTACK. Hold task/optimizer/protocol fixed; scale capacity
    (d_model, N) at constant overload (K=2N). Does bimodality survive, weaken, or dissolve?"""
    sc = mp["scale"]
    points = list(sc["points"])             # each {N, d_model, K}
    n_seeds = int(sc.get("n_seeds", 12))
    steps = int(sc.get("steps", 30000))
    arch, workers = mp.get("arch", "diag_ssm"), int(mp.get("workers", 3))
    try:                                          # spread cells across all visible GPUs (Kaggle T4 x2)
        import torch; ngpus = torch.cuda.device_count()
    except Exception:
        ngpus = 1
    runs_dir = cfg.data_path("runs") / "mp0b" / "scale"; runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] SCALE ATTACK: {len(points)} capacity points x {n_seeds} seeds, steps={steps}; gpus={ngpus}.")
    cells = [(i, s) for i in range(len(points)) for s in range(n_seeds)]
    done = db.done_units(STAGE)

    def unit(i, s): return f"{arch}:scale:p{i}:s{s}"
    def mkargs(c):
        i, s = c; pt = points[i]
        m = {k: v for k, v in mp.items() if k != "scale"}
        m["d_model"] = int(pt["d_model"]); m["steps"] = steps
        # round-robin each cell onto a fixed GPU (deterministic per cell; data/init drawn on CPU
        # so results are byte-identical regardless of which identical T4 runs the cell).
        if ngpus > 1 and str(m.get("device", "auto")) in ("auto", "cuda"):
            m["device"] = f"cuda:{(i * n_seeds + s) % ngpus}"
        return (arch, int(pt["N"]), int(pt["K"]), s, m)
    def record(c, out):
        i, s = c
        # save the FULL dense trajectory (MP0b-14 morphology): resolve the transition shape
        (runs_dir / f"p{i}_s{s}.json").write_text(json.dumps(
            {"point": i, **points[i], "seed": s, "final_k": out["k_correct"], "jump": _jump(out),
             "history": out.get("history", [])},
            indent=1), encoding="utf-8")
        db.mark(STAGE, unit(i, s), "done")
        log.dim(f"[MP0b] scale p{i}{points[i]} s{s}: final_k={out['k_correct']:.1f} jump={_jump(out)}")

    pending = [c for c in cells if unit(*c) not in done]
    for c in pending:
        db.mark(STAGE, unit(*c), "running")
    _pool_run(db, pending, mkargs, record, workers)

    import statistics
    out = {"points": []}
    for i, pt in enumerate(points):
        recs = [json.loads(p.read_text()) for p in runs_dir.glob(f"p{i}_s*.json")]
        if not recs: continue
        fk = sorted(r["final_k"] for r in recs)
        jp = [r["jump"] for r in recs if r["jump"] is not None]
        # bimodality: largest gap relative to range; clusters either side
        gaps = [(fk[j+1]-fk[j], fk[j], fk[j+1]) for j in range(len(fk)-1)]
        biggest = max(gaps, key=lambda g: g[0]) if gaps else (0, 0, 0)
        rng = (fk[-1]-fk[0]) or 1
        gap_frac = biggest[0]/rng
        lower = [x for x in fk if x <= biggest[1]]; upper = [x for x in fk if x >= biggest[2]]
        bimodal = gap_frac > 0.35 and len(lower) >= 2 and len(upper) >= 2
        rec = {"point": pt, "n": len(fk), "finals": [round(x, 1) for x in fk],
               "min": round(fk[0], 1), "max": round(fk[-1], 1), "cv": round(statistics.pstdev(fk)/statistics.mean(fk), 3),
               "largest_gap": round(biggest[0], 1), "gap_frac": round(gap_frac, 2),
               "stuck_frac": round(len(lower)/len(fk), 2), "bimodal": bimodal,
               "jump_min": min(jp) if jp else None, "jump_max": max(jp) if jp else None}
        out["points"].append(rec)
        log.success(f"[MP0b] {pt}: BIMODAL={bimodal} gap_frac={rec['gap_frac']} stuck={rec['stuck_frac']} "
                    f"cv={rec['cv']} finals={rec['finals']}")
    log.info("[MP0b] bimodal at every scale => phenomenon SURVIVES; collapses => toy-scale artifact.")
    return out


def run_data_switch(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-16: critical-window probe. MP0b-15 found that for a BORDERLINE init the recall
    basin is selected MID-TRAINING by the data stream (i3 is stuck under d0/d1/d3 but rescued
    by d2, diverging ~20-50k). Here we switch the training data stream between the rescuing
    seed and a non-rescuing seed at step T and ask WHEN the data is decisive:
      - 'rm_T'  : start rescuing (d2), switch to stuck (d0) at T -> when is the rescue locked in?
      - 'add_T' : start stuck (d0), switch to rescuing (d2) at T -> when does the window close?
    The eval set stays fixed (tied to base data_seed), so finals are comparable within a series."""
    ds = mp["data_switch"]
    init = int(ds.get("init_seed", 3))
    rescue = int(ds.get("rescue_seed", 2))     # data seed that rescues this init (i3xd2)
    stuck = int(ds.get("stuck_seed", 0))       # a non-rescuing data seed for this init
    Ts = list(ds.get("switch_steps", [10000, 20000, 30000, 50000, 75000]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b" / "data_switch"; runs_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[MP0b] DATA-SWITCH probe init={init} rescue=d{rescue} stuck=d{stuck} on {arch} "
             f"N={N} K={K}: switch steps {Ts} (both directions) + 2 controls.")

    cells = [("ctrl_rescue", {"init_seed": init, "data_seed": rescue}),
             ("ctrl_stuck", {"init_seed": init, "data_seed": stuck})]
    for T in Ts:
        cells.append((f"rm_T{T}", {"init_seed": init, "data_seed": rescue,
                                   "data_switch_at": T, "data_switch_seed": stuck}))
        cells.append((f"add_T{T}", {"init_seed": init, "data_seed": stuck,
                                    "data_switch_at": T, "data_switch_seed": rescue}))

    def unit(tag): return f"{arch}:dswitch:N{N}:K{K}:{tag}"
    def mkargs(c):
        tag, over = c
        m = {k: v for k, v in mp.items() if k != "data_switch"}
        m.update(over)
        return (arch, N, K, 0, m)
    def record(c, out):
        tag, _ = c
        (runs_dir / f"{tag}.json").write_text(json.dumps(
            {"tag": tag, "final_k": out["k_correct"], "accuracy": out["accuracy"],
             "jump": _jump(out), "history": out.get("history", [])}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(tag), "done")
        log.dim(f"[MP0b] dswitch {tag}: final_k={out['k_correct']:.1f} jump={_jump(out)}")

    done = db.done_units(STAGE)
    pending = [c for c in cells if unit(c[0]) not in done]
    for c in pending:
        db.mark(STAGE, unit(c[0]), "running")
    _pool_run(db, pending, mkargs, record, workers)

    recs = {json.loads(p.read_text())["tag"]: json.loads(p.read_text())
            for p in runs_dir.glob("*.json")}
    out = {"controls": {}, "rm": [], "add": []}
    for tag in ("ctrl_rescue", "ctrl_stuck"):
        if tag in recs:
            out["controls"][tag] = round(recs[tag]["final_k"], 1)
    for T in Ts:
        if f"rm_T{T}" in recs:
            out["rm"].append({"T": T, "final_k": round(recs[f"rm_T{T}"]["final_k"], 1)})
        if f"add_T{T}" in recs:
            out["add"].append({"T": T, "final_k": round(recs[f"add_T{T}"]["final_k"], 1)})
    log.success(f"[MP0b] dswitch controls: {out['controls']}")
    log.success(f"[MP0b] rm (start rescue d{rescue}, ->d{stuck} at T): "
                + " ".join(f"T{r['T']//1000}k={r['final_k']}" for r in out["rm"]))
    log.success(f"[MP0b] add(start stuck d{stuck}, ->d{rescue} at T): "
                + " ".join(f"T{a['T']//1000}k={a['final_k']}" for a in out["add"]))
    log.info("[MP0b] rm crossover = rescue LOCK-IN step; add crossover = window CLOSE step.")
    return out


def run(db: Database, cfg, run_id: str) -> dict[str, Any]:
    mp = dict(cfg.get("mp0b", default={}) or {})
    if mp.get("data_switch"):
        return run_data_switch(db, cfg, run_id, mp)
    if mp.get("scale"):
        return run_scale(db, cfg, run_id, mp)
    if mp.get("interp"):
        return run_interp(db, cfg, run_id, mp)
    if mp.get("perturb_mag"):
        return run_perturb_mag(db, cfg, run_id, mp)
    if mp.get("perturb_control"):
        return run_perturb_control(db, cfg, run_id, mp)
    if mp.get("hybrid"):
        return run_hybrid(db, cfg, run_id, mp)
    if mp.get("init_predictor"):
        return run_init_predictor(db, cfg, run_id, mp)
    if mp.get("grid2d"):
        return run_grid2d(db, cfg, run_id, mp)
    if mp.get("sweep"):
        return run_sweep(db, cfg, run_id, mp)
    arch = mp.get("arch", "diag_ssm")
    N, K = int(mp.get("N", 8)), int(mp.get("K", 16))
    n_seeds = int(mp.get("n_seeds", 16))
    workers = int(mp.get("workers", 3))
    stuck_frac = float(mp.get("stuck_threshold", 0.7))
    runs_dir = cfg.data_path("runs") / "mp0b"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cells = [(arch, N, K, s) for s in range(n_seeds)]
    done = db.done_units(STAGE)
    pending = [c for c in cells if f"{arch}:N{N}:K{K}:s{c[3]}" not in done]
    log.info(f"[MP0b] Bimodal study of {arch} N={N} K={K}: {n_seeds} seeds; "
             f"{len(done)} done, {len(pending)} to run; workers={workers}.")

    def record(seed: int, out: dict[str, Any]) -> None:
        unit = f"{arch}:N{N}:K{K}:s{seed}"
        (runs_dir / f"s{seed}.json").write_text(json.dumps({
            "seed": seed, "arch": arch, "N": N, "K": K, "final_k": out["k_correct"],
            "accuracy": out["accuracy"], "steps": out["steps"],
            "max_grad_norm": out.get("max_grad_norm"), "history": out.get("history", []),
        }, indent=1), encoding="utf-8")
        db.insert("measurements", {
            "meas_id": "meas|" + short_hash([MP_ID, "bimodal_final_k", arch, N, K, seed]),
            "quantity": "bimodal_final_k", "arch": arch, "dataset": "mqar",
            "N": N, "d_state": N, "K": K, "seed": seed, "value": out["k_correct"],
            "method": MP_ID, "mp_id": MP_ID, "run_id": run_id, "created_at": utcnow(),
        })
        db.mark(STAGE, unit, "done")
        log.dim(f"[MP0b] seed {seed}: final_k={out['k_correct']:.1f} acc={out['accuracy']:.3f}")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, (*c, mp)): c for c in pending}
            for c in pending:
                db.mark(STAGE, f"{arch}:N{N}:K{K}:s{c[3]}", "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    seed = futs[fut][3]
                    try:
                        _, out = fut.result()
                        record(seed, out)
                    except Exception as exc:
                        db.mark(STAGE, f"{arch}:N{N}:K{K}:s{seed}", "failed", str(exc))
                        log.warn(f"[MP0b] seed {seed} failed: {exc}")
    else:
        for c in pending:
            _, out = _train_worker((*c, mp))
            record(c[3], out)

    return analyze(db, cfg, arch, N, K, stuck_frac, run_id)


def run_sweep(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-3+: vary ONE factor across groups of seeds; measure how it shifts the
    jump-time distribution / stuck fraction (does it govern the metastable transition?)."""
    sw = mp["sweep"]
    param = sw["param"]
    values = list(sw["values"])
    n_seeds = int(sw.get("seeds", 8))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b"
    log.info(f"[MP0b] SWEEP {param}={values} x {n_seeds} seeds on {arch} N={N} K={K}; workers={workers}.")

    def unit(v, s):
        return f"{arch}:N{N}:K{K}:{param}{v}:s{s}"

    def args(v, s):
        m = dict(mp); m[param] = v; m.pop("sweep", None)
        return (arch, N, K, s, m)

    done = db.done_units(STAGE)
    pending = [(v, s) for v in values for s in range(n_seeds) if unit(v, s) not in done]

    def record(v, s, out):
        sub = runs_dir / f"{param}_{v}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"s{s}.json").write_text(json.dumps({
            "seed": s, param: v, "final_k": out["k_correct"], "accuracy": out["accuracy"],
            "history": out.get("history", [])}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(v, s), "done")
        log.dim(f"[MP0b] {param}={v} seed {s}: final_k={out['k_correct']:.1f}")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, args(v, s)): (v, s) for v, s in pending}
            for v, s in pending:
                db.mark(STAGE, unit(v, s), "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    v, s = futs[fut]
                    try:
                        _, out = fut.result()
                        record(v, s, out)
                    except Exception as exc:
                        db.mark(STAGE, unit(v, s), "failed", str(exc))
                        log.warn(f"[MP0b] {param}={v} seed {s} failed: {exc}")
    else:
        for v, s in pending:
            _, out = _train_worker(args(v, s))
            record(v, s, out)

    # analyze per value: stuck fraction + jump-time distribution
    summary = {"param": param, "groups": []}
    for v in values:
        sub = runs_dir / f"{param}_{v}"
        finals, jumps = [], []
        for p in sorted(sub.glob("s*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            finals.append(d["final_k"])
            js = next((h["step"] for h in (d.get("history") or []) if h.get("eval_acc", 0) >= 0.6), None)
            jumps.append(js)
        if not finals:
            continue
        hi = max(finals)
        n_stuck = sum(1 for f in finals if f < 0.7 * hi)
        jumped = [j for j in jumps if j is not None]
        g = {param: v, "n": len(finals), "finals": [round(f, 1) for f in sorted(finals)],
             "frac_stuck": round(n_stuck / len(finals), 2), "n_jumped": len(jumped),
             "median_jump": int(statistics.median(jumped)) if jumped else None}
        summary["groups"].append(g)
        log.success(f"[MP0b] {param}={v}: frac_stuck={g['frac_stuck']} "
                    f"jumped={g['n_jumped']}/{g['n']} median_jump={g['median_jump']} finals={g['finals']}")
    return summary


def run_grid2d(db: Database, cfg, run_id: str, mp: dict[str, Any]) -> dict[str, Any]:
    """MP0b-5: decouple init_seed x data_seed. Does a seed's fate track its INIT
    (lottery-ticket / init-basin) or its DATA order (SGD trajectory)?"""
    g2 = mp["grid2d"]
    inits = list(g2.get("init_seeds", [0, 1, 2, 3]))
    datas = list(g2.get("data_seeds", [0, 1, 2, 3]))
    arch, N, K = mp.get("arch", "diag_ssm"), int(mp["N"]), int(mp["K"])
    workers = int(mp.get("workers", 3))
    runs_dir = cfg.data_path("runs") / "mp0b"
    log.info(f"[MP0b] GRID2D init x data = {len(inits)}x{len(datas)} on {arch} N={N} K={K}.")

    def unit(i, d):
        return f"{arch}:N{N}:K{K}:i{i}:d{d}"

    def args(i, d):
        m = dict(mp); m["init_seed"] = i; m["data_seed"] = d; m.pop("grid2d", None)
        return (arch, N, K, 0, m)   # seed arg unused once init/data seeds are set

    done = db.done_units(STAGE)
    pending = [(i, d) for i in inits for d in datas if unit(i, d) not in done]

    def record(i, d, out):
        sub = runs_dir / "grid2d"
        sub.mkdir(parents=True, exist_ok=True)
        # MP0b-15: save the dense trajectory too, so ramp RATE (not just final_k) can be
        # compared across data seeds within an init row (does init pin the rate?).
        (sub / f"i{i}_d{d}.json").write_text(json.dumps({
            "init_seed": i, "data_seed": d, "final_k": out["k_correct"],
            "accuracy": out["accuracy"], "jump": _jump(out),
            "history": out.get("history", [])}, indent=1), encoding="utf-8")
        db.mark(STAGE, unit(i, d), "done")
        log.dim(f"[MP0b] init={i} data={d}: final_k={out['k_correct']:.1f}")

    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_worker, args(i, d)): (i, d) for i, d in pending}
            for i, d in pending:
                db.mark(STAGE, unit(i, d), "running")
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    i, d = futs[fut]
                    try:
                        _, out = fut.result()
                        record(i, d, out)
                    except Exception as exc:
                        db.mark(STAGE, unit(i, d), "failed", str(exc))
                        log.warn(f"[MP0b] init={i} data={d} failed: {exc}")
    else:
        for i, d in pending:
            _, out = _train_worker(args(i, d))
            record(i, d, out)

    return _analyze_grid2d(cfg, inits, datas)


def _analyze_grid2d(cfg, inits, datas) -> dict[str, Any]:
    sub = cfg.data_path("runs") / "mp0b" / "grid2d"
    table = {}
    for p in sub.glob("i*_d*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        table[(int(d["init_seed"]), int(d["data_seed"]))] = d["final_k"]
    if not table:
        return {}
    # print the init x data table
    log.success("[MP0b] final_k table (rows=init_seed, cols=data_seed):")
    header = "      " + "  ".join(f"d{dd}" for dd in datas)
    log.info(header)
    for i in inits:
        row = "  ".join(f"{table.get((i, dd), float('nan')):4.1f}" for dd in datas)
        log.info(f"  i{i}  {row}")
    # variance decomposition: is final_k explained by init (rows) or data (cols)?
    vals = list(table.values())
    grand = statistics.mean(vals)
    row_means = {i: statistics.mean([table[(i, d)] for d in datas if (i, d) in table]) for i in inits}
    col_means = {d: statistics.mean([table[(i, d)] for i in inits if (i, d) in table]) for d in datas}
    ss_total = sum((v - grand) ** 2 for v in vals)
    ss_init = sum(len(datas) * (row_means[i] - grand) ** 2 for i in inits)
    ss_data = sum(len(inits) * (col_means[d] - grand) ** 2 for d in datas)
    res = {
        "n": len(table), "grand_mean": round(grand, 2),
        "var_explained_by_init": round(ss_init / ss_total, 3) if ss_total else None,
        "var_explained_by_data": round(ss_data / ss_total, 3) if ss_total else None,
        "row_means_init": {i: round(m, 1) for i, m in row_means.items()},
        "col_means_data": {d: round(m, 1) for d, m in col_means.items()},
    }
    verdict = ("INIT-determined (lottery-ticket-like)" if res["var_explained_by_init"] and
               res["var_explained_by_init"] > 2 * (res["var_explained_by_data"] or 0)
               else "DATA-determined (SGD path)" if res["var_explained_by_data"] and
               res["var_explained_by_data"] > 2 * (res["var_explained_by_init"] or 0)
               else "neither dominates (interaction / both)")
    res["verdict"] = verdict
    log.success(f"[MP0b] var by init={res['var_explained_by_init']} "
                f"var by data={res['var_explained_by_data']} -> {verdict}")
    return res


def analyze(db: Database, cfg, arch: str, N: int, K: int, stuck_frac: float, run_id: str) -> dict[str, Any]:
    rows = db.query(
        "SELECT seed, value FROM measurements WHERE quantity='bimodal_final_k' "
        "AND arch=? AND N=? AND K=? ORDER BY value", (arch, N, K))
    finals = [float(r["value"]) for r in rows]
    if len(finals) < 2:
        return {"n": len(finals)}
    lo, hi = min(finals), max(finals)
    threshold = stuck_frac * hi
    stuck = [v for v in finals if v < threshold]
    plateau = [v for v in finals if v >= threshold]
    # largest gap between sorted finals -> evidence of two clusters
    gaps = [(finals[i + 1] - finals[i], finals[i], finals[i + 1]) for i in range(len(finals) - 1)]
    biggest = max(gaps, key=lambda g: g[0]) if gaps else (0, 0, 0)

    # metastability: from each seed's trajectory, the step at which eval_acc first
    # reaches 90% of that seed's own final acc (the "jump" step)
    jumps = _jump_steps(cfg, arch, N, K)

    res = {
        "arch": arch, "N": N, "K": K, "n_seeds": len(finals),
        "finals_sorted": [round(v, 1) for v in finals],
        "min": round(lo, 1), "max": round(hi, 1),
        "mean": round(statistics.mean(finals), 2),
        "stdev": round(statistics.pstdev(finals), 2),
        "cv": round(statistics.pstdev(finals) / statistics.mean(finals), 3) if statistics.mean(finals) else None,
        "n_stuck": len(stuck), "n_plateau": len(plateau),
        "frac_stuck": round(len(stuck) / len(finals), 2),
        "largest_gap": round(biggest[0], 1), "gap_between": [round(biggest[1], 1), round(biggest[2], 1)],
        "jump_steps": jumps,
    }
    log.success(f"[MP0b] finals(sorted)={res['finals_sorted']}")
    log.info(f"[MP0b] n={res['n_seeds']} mean={res['mean']} cv={res['cv']} "
             f"stuck={res['n_stuck']}/{res['n_seeds']} (frac {res['frac_stuck']}) "
             f"largest_gap={res['largest_gap']} between {res['gap_between']}")
    if jumps:
        log.info(f"[MP0b] jump steps: plateau={jumps.get('plateau')} stuck={jumps.get('stuck')}")

    # record/refresh the phenomenon with the measured distribution
    ph.record(db, ph.Phenomenon(
        name="bimodal convergence", status="observed",
        statement=(f"[MP0b-1] {arch} N={N} K={K}, {res['n_seeds']} seeds, identical hyperparams: "
                   f"finals {res['finals_sorted']}; {res['n_stuck']}/{res['n_seeds']} stuck-low "
                   f"(frac {res['frac_stuck']}), CV {res['cv']}, largest gap {res['largest_gap']} "
                   f"between {res['gap_between']}. Same architecture -> two worlds."),
        support=[f"MP0b-1 {arch} N{N} K{K} x{res['n_seeds']} seeds"]), run_id)
    return res


def _jump_steps(cfg, arch: str, N: int, K: int) -> dict[str, Any]:
    """Per-seed step at which eval_acc first reaches 90% of that seed's final acc.
    Separated into plateau vs stuck (by final_k median) to test for grokking-like delay."""
    runs_dir = cfg.data_path("runs") / "mp0b"
    seeds = []
    for p in sorted(runs_dir.glob("s*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        hist = d.get("history") or []
        if not hist:
            continue
        final_acc = d.get("accuracy", 0.0)
        target = 0.9 * final_acc
        jstep = next((h["step"] for h in hist if h.get("eval_acc", 0) >= target and target > 0), None)
        seeds.append({"final_k": d.get("final_k", 0.0), "jump": jstep})
    if not seeds:
        return {}
    med = statistics.median(s["final_k"] for s in seeds)
    plateau = [s["jump"] for s in seeds if s["final_k"] >= med and s["jump"] is not None]
    stuck = [s["jump"] for s in seeds if s["final_k"] < med and s["jump"] is not None]
    return {
        "plateau": {"n": len(plateau), "median_jump": int(statistics.median(plateau)) if plateau else None},
        "stuck": {"n": len(stuck), "median_jump": int(statistics.median(stuck)) if stuck else None},
    }
