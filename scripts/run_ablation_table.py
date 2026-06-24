#!/usr/bin/env python3
"""Faithful L2 ablation table on the REAL SymbolicExecutor.

For each instance (increasing size) and seed, run the real executor in two modes:
  FULL      : pheromone-weighted roulette selection (the deployed system)
  ABLATION  : random operator selection (ablation_mode=True; no pheromones/rules)
              == the paper's "random ordering" ablation (Sec. 1544, Random Search).

Reuses scripts/run_symbolic_tsp.py (the real component, fixed imports) so nothing
is reconstructed. Reports mean +/- std gap over seeds; the size-dependent pattern
(small instances: full == random; larger: full wins) is the expected signal.

  uv run python scripts/run_ablation_table.py \
    --instances berlin52 kroA100 ch150 pr226 pcb442 --seeds 42 43 44 --time 6
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


# TSPLIB known optima (fallback if not embedded in the .tsp file)
OPT = {"berlin52": 7542, "kroA100": 21282, "ch150": 6528, "pr226": 80369,
       "pcb442": 50778, "rat783": 8806, "kroA200": 29368, "pr299": 48191}


def run_one(rst, dm, snapshot, pool, n, opt, ablation, seed, time_limit):
    """Single SUSTAINED execute (no multistart). The multistart wrapper resets the
    rule engine every ~time/10, which forces FULL mode back into its CONSTRUCTION
    phase each restart (re-shuffling the tour) and unfairly penalizes it vs random.
    A single sustained run isolates the actual question: does pheromone-guided
    selection produce better operator sequences than random?"""
    random.seed(seed)
    ctx = rst.TSPContext(dm)
    rule_engine = rst.extract_symbolic_rules(snapshot)   # stateful: rebuild per run
    operators = rst.compile_operators(pool, ctx)
    pher = snapshot.get("pheromones", {}).get("operator_level", {})
    ex = rst.SymbolicExecutor(
        rule_engine=rule_engine, evaluate_fn=lambda s, i: ctx.evaluate(s),
        copy_fn=lambda s: s[:], operator_pheromones=pher, global_mode=False,
        ablation_mode=ablation, verbose=False,
    )
    init = list(range(n))
    random.shuffle(init)
    res = ex.execute(operators=operators, initial_solution=init,
                     initial_cost=ctx.evaluate(init), time_limit=time_limit, instance=None)
    return 100.0 * (res.best_cost - opt) / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+",
                    default=["berlin52", "kroA100", "ch150", "pr226", "pcb442"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--time", type=float, default=6.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--output", default="results/ablation/l2_table.json")
    args = ap.parse_args()

    rst = load_rst()
    snapshot = rst.load_snapshot(Path(args.snapshot))
    pool = rst.load_pool(Path(args.snapshot).parent / "refined_pool.json")

    print(f"snapshot={args.snapshot}")
    print(f"seeds={args.seeds}  time={args.time}s  (FULL=pheromone-weighted, "
          f"ABLATION=random)\n")
    print(f"{'instance':10} {'n':>5} {'FULL gap%':>16} {'ABLATION gap%':>16} {'Δ abl-full':>12}")
    print("-" * 64)
    rows = []
    for name in args.instances:
        path = f"data/instances/tsp/{name}.tsp"
        dm, opt = rst.load_tsplib(path)
        opt = opt or OPT.get(name)
        n = len(dm)
        full = [run_one(rst, dm, snapshot, pool, n, opt, False, s, args.time) for s in args.seeds]
        abl = [run_one(rst, dm, snapshot, pool, n, opt, True, s, args.time) for s in args.seeds]
        fm = statistics.mean(full); fs = statistics.pstdev(full) if len(full) > 1 else 0.0
        am = statistics.mean(abl); as_ = statistics.pstdev(abl) if len(abl) > 1 else 0.0
        fmed = statistics.median(full); amed = statistics.median(abl)
        print(f"{name:10} {n:>5}  full {fm:>6.2f}±{fs:<5.2f} med{fmed:>6.2f}   "
              f"abl {am:>7.2f}±{as_:<6.2f} med{amed:>7.2f}   Δmed {amed - fmed:>+7.2f}")
        rows.append({"instance": name, "n": n, "opt": opt,
                     "full_mean": fm, "full_std": fs, "full_median": fmed, "full": full,
                     "ablation_mean": am, "ablation_std": as_, "ablation_median": amed,
                     "ablation": abl, "delta_mean": am - fm, "delta_median": amed - fmed})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"snapshot": args.snapshot, "seeds": args.seeds,
                   "time": args.time, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
