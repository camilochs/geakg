#!/usr/bin/env python3
"""Anytime convergence curve: FULL (pheromone-weighted) vs ABLATION (random),
averaged over many seeds, on the REAL SymbolicExecutor.

Rationale: a final-best-gap at wall-clock is noisy and lets a strong randomized
local search tie. The paper's clean L2 evidence (NAS) uses a FIXED EVALUATION
budget. The analogous TSP test is sample efficiency: gap reached as a function of
search progress (time/iterations). Averaging the anytime curve over many seeds
suppresses the per-run noise; if pheromone guidance helps, FULL's mean curve sits
below ABLATION's (lower gap sooner).

  uv run python scripts/run_anytime_curve.py --instance ch150 --opt 6528 --seeds 15 --time 15
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path


def load_rst():
    spec = importlib.util.spec_from_file_location("rst", "scripts/run_symbolic_tsp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gap_curve(history, initial_cost, opt, checkpoints):
    """best gap (%) at each time checkpoint, from improvement history (monotone)."""
    evs = sorted((h["time"], h["cost"]) for h in history if h.get("event") == "improvement")
    out = []
    for t in checkpoints:
        c = initial_cost
        for ht, hc in evs:
            if ht <= t:
                c = hc
            else:
                break
        out.append(100.0 * (c - opt) / opt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="ch150")
    ap.add_argument("--opt", type=float, default=6528)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--time", type=float, default=15.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--output", default="results/ablation/anytime_curve.json")
    args = ap.parse_args()

    rst = load_rst()
    snap = rst.load_snapshot(Path(args.snapshot))
    pool = rst.load_pool(Path(args.snapshot).parent / "refined_pool.json")
    pher = snap["pheromones"]["operator_level"]
    dm, opt = rst.load_tsplib(f"data/instances/tsp/{args.instance}.tsp")
    opt = opt or args.opt
    n = len(dm)

    checkpoints = [0.5, 1, 2, 3, 5, 7, 10, args.time]
    curves = {"full": [], "ablation": []}
    for ablation in (False, True):
        key = "ablation" if ablation else "full"
        for seed in range(args.seeds):
            random.seed(1000 + seed)
            ctx = rst.TSPContext(dm)
            re = rst.extract_symbolic_rules(snap)
            ops = rst.compile_operators(pool, ctx)
            ex = rst.SymbolicExecutor(
                rule_engine=re, evaluate_fn=lambda s, i: ctx.evaluate(s),
                copy_fn=lambda s: s[:], operator_pheromones=pher,
                global_mode=False, ablation_mode=ablation, verbose=False)
            init = list(range(n)); random.shuffle(init)
            res = ex.execute(operators=ops, initial_solution=init,
                             initial_cost=ctx.evaluate(init), time_limit=args.time, instance=None)
            curves[key].append(gap_curve(res.history, ctx.evaluate(init), opt, checkpoints))

    print(f"{args.instance} (n={n}, opt={opt})  seeds={args.seeds}  t={args.time}s")
    print(f"mean gap%% at each time checkpoint (lower=better)\n")
    print(f"{'t(s)':>6} {'FULL gap%':>12} {'ABL gap%':>12} {'Δ(abl-full)':>12}")
    print("-" * 46)
    summary = []
    for i, t in enumerate(checkpoints):
        fm = statistics.mean(c[i] for c in curves["full"])
        am = statistics.mean(c[i] for c in curves["ablation"])
        print(f"{t:>6.1f} {fm:>12.3f} {am:>12.3f} {am - fm:>+12.3f}")
        summary.append({"t": t, "full_mean": fm, "ablation_mean": am, "delta": am - fm})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"instance": args.instance, "n": n, "opt": opt, "seeds": args.seeds,
                   "checkpoints": checkpoints, "summary": summary,
                   "full_curves": curves["full"], "ablation_curves": curves["ablation"]}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
