"""MP0b-15 analyzer (read-only) — init x data decoupling at N=16/150k.

Reads data/runs/mp0b/grid2d/i*_d*.json (written by run_grid2d) and reports:
  1. determinism gate: diagonal cells (i==d) must reproduce the scale run;
  2. the init x data final_k table;
  3. variance decomposition (init vs data vs interaction) -- same math as _analyze_grid2d;
  4. ramp-rate-by-init from the saved trajectories (does init pin the RATE, not just the final?);
  5. the verdict per the MP0b-15 decision table.

Safe to run while `cde study` is still going: it only reads finished-cell JSONs.
Usage:  python scripts/mp0b15_analyze.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "runs" / "mp0b" / "grid2d"

# Reference finals from the coupled N=16/150k scale run (init==data==s). The diagonal
# grid cells must byte-reproduce these if determinism holds.
SCALE_REF = {0: 7.6875, 1: 22.34375, 2: 19.046875, 3: 11.578125}
TOL = 1e-3  # determinism should be exact; allow float-repr slack only


def load():
    cells = {}
    if not GRID.exists():
        return cells
    for p in sorted(GRID.glob("i*_d*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ! unreadable {p.name}: {exc}")
            continue
        cells[(int(d["init_seed"]), int(d["data_seed"]))] = d
    return cells


def ramp_metrics(hist):
    """Summarise a trajectory: final acc, step to cross 0.6, mean gain over the last 20% of steps."""
    if not hist:
        return None
    accs = [(h["step"], h.get("eval_acc", 0.0)) for h in hist]
    final = accs[-1][1]
    cross = next((s for s, a in accs if a >= 0.6), None)
    tail = accs[int(0.8 * len(accs)):]
    slope = None
    if len(tail) >= 2 and tail[-1][0] != tail[0][0]:
        slope = (tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0]) * 1000  # acc gain / 1k steps
    return {"final_acc": final, "cross_0.6": cross, "tail_slope_per_1k": slope}


def main():
    cells = load()
    inits = sorted({i for i, _ in cells})
    datas = sorted({d for _, d in cells})
    print(f"MP0b-15 grid2d  ({len(cells)}/16 cells present)  inits={inits} datas={datas}\n")
    if not cells:
        print("No cells yet. (First 150k-step cell lands ~50 min in.)")
        return

    # 1. DETERMINISM GATE -------------------------------------------------
    print("== DETERMINISM GATE (diagonal i==d vs scale run) ==")
    gate_ok = True
    for s in inits:
        if (s, s) in cells and s in SCALE_REF:
            got = cells[(s, s)]["final_k"]
            ref = SCALE_REF[s]
            ok = abs(got - ref) <= TOL
            gate_ok &= ok
            print(f"  i{s}_d{s}: final_k={got:.5f}  ref={ref:.5f}  {'OK' if ok else '*** MISMATCH ***'}")
        elif s in SCALE_REF:
            print(f"  i{s}_d{s}: (not done yet, ref={SCALE_REF[s]})")
    print(f"  -> gate: {'PASS (determinism holds)' if gate_ok else 'FAIL — HALT, do not interpret'}\n")

    # 2. TABLE ------------------------------------------------------------
    print("== final_k table (rows=init, cols=data) ==")
    print("       " + "  ".join(f"d{d:>4}" for d in datas))
    for i in inits:
        row = "  ".join(f"{cells[(i, d)]['final_k']:5.1f}" if (i, d) in cells else "  .  " for d in datas)
        print(f"  i{i}   {row}")
    print()

    # 3. VARIANCE DECOMPOSITION (only on the complete sub-grid) -----------
    full = [(i, d) for i in inits for d in datas if (i, d) in cells]
    if len(full) == len(inits) * len(datas) and len(inits) >= 2 and len(datas) >= 2:
        vals = {(i, d): cells[(i, d)]["final_k"] for i, d in full}
        grand = statistics.mean(vals.values())
        row_m = {i: statistics.mean([vals[(i, d)] for d in datas]) for i in inits}
        col_m = {d: statistics.mean([vals[(i, d)] for i in inits]) for d in datas}
        ss_total = sum((v - grand) ** 2 for v in vals.values())
        ss_init = sum(len(datas) * (row_m[i] - grand) ** 2 for i in inits)
        ss_data = sum(len(inits) * (col_m[d] - grand) ** 2 for d in datas)
        f_init = ss_init / ss_total if ss_total else 0
        f_data = ss_data / ss_total if ss_total else 0
        f_inter = max(0.0, 1 - f_init - f_data)
        print("== variance decomposition ==")
        print(f"  var by INIT (rows): {f_init:.3f}")
        print(f"  var by DATA (cols): {f_data:.3f}")
        print(f"  interaction/resid : {f_inter:.3f}")
        print(f"  row means (init): { {i: round(m,1) for i,m in row_m.items()} }")
        print(f"  col means (data): { {d: round(m,1) for d,m in col_m.items()} }")
        # 5. VERDICT
        if f_init > 2 * f_data:
            v = "INIT-DETERMINED survives at N=16 (re-justifies coordinate hunt; wounds L1: b seed-fragile)"
        elif f_data > 2 * f_init:
            v = "DATA-DETERMINED -> 'init-determined' DIES; ramp rate is an SGD/data-path property"
        elif f_inter > max(f_init, f_data):
            v = "INTERACTION-DETERMINED -> neither factor alone; irreducible seed-pairing (most deflationary)"
        else:
            v = "MIXED / no factor dominates -> report split honestly, no clean lottery-ticket reading"
        print(f"\n  VERDICT: {v}\n")
    else:
        print(f"== variance decomposition: waiting for full grid "
              f"({len(full)}/{len(inits)*len(datas)} cells) ==\n")

    # 4. RAMP-RATE-BY-INIT -----------------------------------------------
    print("== ramp metrics per cell (final_acc | cross0.6 | tail slope /1k) ==")
    for i in inits:
        parts = []
        for d in datas:
            if (i, d) not in cells:
                parts.append(f"d{d}: .")
                continue
            rm = ramp_metrics(cells[(i, d)].get("history"))
            if rm is None:
                parts.append(f"d{d}: (no hist)")
            else:
                c = rm["cross_0.6"]
                sl = rm["tail_slope_per_1k"]
                parts.append(f"d{d}: acc={rm['final_acc']:.2f} cross={c if c else '--'} "
                             f"slope={sl:.3f}" if sl is not None else
                             f"d{d}: acc={rm['final_acc']:.2f} cross={c if c else '--'} slope=--")
        print(f"  i{i}:  " + "   ".join(parts))
    print("\n  (If init pins the RATE: rows are homogeneous in slope/cross across data seeds.)")


if __name__ == "__main__":
    main()
