#!/usr/bin/env python3
"""Clean L2 ablation: LEARNED pheromones vs UNIFORM pheromones, holding the full
rule/phase structure fixed (ablation_mode=False in both arms).

This isolates exactly what L2 learns -- the operator weighting -- without the
confound of `ablation_mode` (which also strips the rule structure and uses global
random selection, i.e. a different algorithm). Both arms use the identical
SymbolicExecutor + rules + phases; only the operator_pheromones differ:
  LEARNED  : snapshot.pheromones.operator_level (what ACO learned)
  UNIFORM  : all weights equal (no learned preference)

If LEARNED < UNIFORM (lower gap), the learned transition preferences help.

  uv run python scripts/run_pheromone_ablation.py \
    --instances berlin52 kroA100 ch150 pr226 --seeds 10 --time 12
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


OPT = {"berlin52": 7542, "kroA100": 21282, "ch150": 6528, "pr226": 80369}


def run_one(rst, dm, snap, pool, n, opt, pher, seed, time_limit):
    random.seed(seed)
    ctx = rst.TSPContext(dm)
    re = rst.extract_symbolic_rules(snap)
    ops = rst.compile_operators(pool, ctx)
    ex = rst.SymbolicExecutor(
        rule_engine=re, evaluate_fn=lambda s, i: ctx.evaluate(s),
        copy_fn=lambda s: s[:], operator_pheromones=pher,
        global_mode=False, ablation_mode=False, verbose=False)  # FULL rules in both arms
    init = list(range(n)); random.shuffle(init)
    res = ex.execute(operators=ops, initial_solution=init,
                     initial_cost=ctx.evaluate(init), time_limit=time_limit, instance=None)
    return 100.0 * (res.best_cost - opt) / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["berlin52", "kroA100", "ch150", "pr226"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--time", type=float, default=12.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--output", default="results/ablation/pheromone_ablation.json")
    args = ap.parse_args()

    rst = load_rst()
    snap = rst.load_snapshot(Path(args.snapshot))
    pool = rst.load_pool(Path(args.snapshot).parent / "refined_pool.json")
    learned = snap["pheromones"]["operator_level"]
    uniform = {k: 1.0 for k in learned}   # same keys, equal weights

    print(f"seeds={args.seeds}  time={args.time}s   (FULL rules in both; only pheromones differ)\n")
    print(f"{'inst':9}{'n':>5}  {'LEARNED med':>14} {'UNIFORM med':>14}  {'Δmed(unif-learn)':>16}")
    print("-" * 60)
    rows = []
    for name in args.instances:
        dm, opt = rst.load_tsplib(f"data/instances/tsp/{name}.tsp")
        opt = opt or OPT.get(name)
        n = len(dm)
        learn = [run_one(rst, dm, snap, pool, n, opt, learned, 100 + s, args.time) for s in range(args.seeds)]
        unif = [run_one(rst, dm, snap, pool, n, opt, uniform, 100 + s, args.time) for s in range(args.seeds)]
        lmed = statistics.median(learn); umed = statistics.median(unif)
        lm = statistics.mean(learn); um = statistics.mean(unif)
        ls = statistics.pstdev(learn) if len(learn) > 1 else 0.0
        us = statistics.pstdev(unif) if len(unif) > 1 else 0.0
        # one-sided Wilcoxon/Mann-Whitney: is LEARNED < UNIFORM? (paper uses Wilcoxon)
        try:
            from scipy.stats import mannwhitneyu
            p = mannwhitneyu(learn, unif, alternative="less").pvalue
        except Exception:
            p = float("nan")
        sig = "*" if p < 0.05 else " "
        verdict = "LEARNED helps" if umed - lmed > 0.05 else ("LEARNED hurts" if umed - lmed < -0.05 else "tie")
        print(f"{name:9}{n:>5}  {lmed:>7.2f}(μ{lm:.2f}±{ls:.2f}) {umed:>7.2f}(μ{um:.2f}±{us:.2f})  "
              f"{umed-lmed:>+8.2f}  p={p:.3f}{sig}  {verdict}")
        rows.append({"instance": name, "n": n, "opt": opt,
                     "learned_median": lmed, "learned_mean": lm, "learned_std": ls, "learned": learn,
                     "uniform_median": umed, "uniform_mean": um, "uniform_std": us, "uniform": unif,
                     "delta_median": umed - lmed, "p_value": p})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"snapshot": args.snapshot, "seeds": args.seeds, "time": args.time, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
