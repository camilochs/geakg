#!/usr/bin/env python3
"""L1 operator-quality ablation / robustness (R1.5; isolates the L1 layer).

Holds L0 (roles, topology) and L2 (learned pheromones, rules, phases) FIXED and
degrades a controlled fraction p of the L1 operator *bodies*. Two degradation modes:

  dead  : the operator becomes a no-op (returns the solution unchanged). Isolates
          whether L1 operator CONTENT is load-bearing. At p=1 every node still gets
          selected with its learned preference, but nothing transforms the solution,
          so only the multistart initial solutions survive.
  noisy : the operator becomes a random 2-swap -- a valid but role-agnostic move.
          This is the R1.5 robustness test (low-quality / adversarial operators).

p=0 reproduces the intact learned snapshot. The SAME real SymbolicExecutor, the SAME
pheromones, and the SAME symbolic rules are used in every cell; only the operator
functions change. Corrupted operators retain their node identity and pheromone weight,
so the learned L2 still routes traffic to them -- that is what makes this an L1
isolation rather than an L0/L2 change.

  uv run python scripts/run_l1_robustness.py \
      --instances berlin52 kroA100 ch150 pr226 --seeds 8 --time 10 \
      --levels 0.0 0.25 0.5 0.75 1.0
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


def corrupt_operators(ops, frac, mode, rng, counter):
    """Replace a random `frac` of operator bodies in place; return how many."""
    n_corrupt = int(round(frac * len(ops)))
    idx = rng.sample(range(len(ops)), n_corrupt) if n_corrupt else []
    for i in idx:
        nm = ops[i].name
        if mode == "dead":
            def make_dead(name):
                def fn(solution, instance=None):
                    counter[name] = counter.get(name, 0) + 1
                    return solution[:]                      # no-op: unchanged
                return fn
            ops[i]._fn = make_dead(nm)
        else:                                               # noisy: random 2-swap
            def make_noisy(name):
                def fn(solution, instance=None):
                    counter[name] = counter.get(name, 0) + 1
                    s = solution[:]
                    if len(s) >= 2:
                        a, b = random.sample(range(len(s)), 2)
                        s[a], s[b] = s[b], s[a]
                    return s
                return fn
            ops[i]._fn = make_noisy(nm)
    return n_corrupt


def run_one(rst, dm, snap, pool, n, opt, pher, frac, mode, seed, time_limit):
    random.seed(seed)
    rng = random.Random(seed)                               # which ops are corrupted
    ctx = rst.TSPContext(dm)
    re = rst.extract_symbolic_rules(snap)
    ops = rst.compile_operators(pool, ctx)
    counter = {}
    n_corrupt = corrupt_operators(ops, frac, mode, rng, counter)
    ex = rst.SymbolicExecutor(
        rule_engine=re, evaluate_fn=lambda s, i: ctx.evaluate(s),
        copy_fn=lambda s: s[:], operator_pheromones=pher,
        global_mode=False, ablation_mode=False, verbose=False)
    init = list(range(n)); random.shuffle(init)
    res = ex.execute(operators=ops, initial_solution=init,
                     initial_cost=ctx.evaluate(init), time_limit=time_limit, instance=None)
    gap = 100.0 * (res.best_cost - opt) / opt
    return gap, n_corrupt, len(ops), sum(counter.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["berlin52", "kroA100", "ch150", "pr226"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--output", default="results/ablation/l1_robustness.json")
    args = ap.parse_args()

    rst = load_rst()
    snap = rst.load_snapshot(Path(args.snapshot))
    pool = rst.load_pool(Path(args.snapshot).parent / "refined_pool.json")
    learned = snap["pheromones"]["operator_level"]

    rows = []
    print(f"L1 robustness | seeds={args.seeds} time={args.time}s | dead(no-op) & noisy(random-swap)\n")
    for name in args.instances:
        dm, opt = rst.load_tsplib(f"data/instances/tsp/{name}.tsp")
        opt = opt or OPT.get(name)
        n = len(dm)
        print(f"--- {name} (n={n}) ---")
        for mode in ("dead", "noisy"):
            for frac in args.levels:
                if frac == 0.0 and mode == "noisy":
                    continue                                # p=0 identical across modes
                gaps, ncs, ntot, ninv = [], 0, 0, 0
                for s in range(args.seeds):
                    g, nc, nt, inv = run_one(rst, dm, snap, pool, n, opt, learned,
                                             frac, mode, 100 + s, args.time)
                    gaps.append(g); ncs, ntot = nc, nt; ninv += inv
                m = statistics.mean(gaps)
                sd = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
                tag = "intact" if frac == 0.0 else mode
                print(f"  {tag:6} p={frac:.2f}  corrupt={ncs:2}/{ntot:2}  "
                      f"gap {m:6.2f}±{sd:4.2f}  (corrupt-calls≈{ninv // max(1, args.seeds)}/run)")
                rows.append({"instance": name, "n": n, "mode": tag, "frac": frac,
                             "n_corrupt": ncs, "n_total": ntot,
                             "gap_mean": m, "gap_std": sd, "gaps": gaps,
                             "mean_corrupt_calls": ninv / args.seeds})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"snapshot": args.snapshot, "seeds": args.seeds, "time": args.time,
                   "rows": rows}, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
