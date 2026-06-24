#!/usr/bin/env python3
"""Bifurcation / perturbation-sensitivity (R4.10): do small perturbations of the
learned pheromones cause unstable trajectories?

Holding the execution RNG fixed, we perturb every learned operator pheromone
multiplicatively, tau' = clip(tau * (1 + eps * N(0,1)), tau_min, tau_max), and run
the REAL Symbolic Executor on TSP. For each perturbation magnitude eps we draw
several random perturbations and measure how much the output gap moves from the
unperturbed baseline. A smooth, bounded response (small eps -> small change) means
the learned policy is stable (no bifurcation), consistent with the spectral gap.
Token-free.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from run_pheromone_ablation import load_rst, OPT, run_one


def perturb(learned, eps, rng, tmin=0.001, tmax=1.0):
    return {k: min(tmax, max(tmin, v * (1 + eps * rng.gauss(0, 1)))) for k, v in learned.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["berlin52", "ch150"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--n-perturb", type=int, default=6)
    ap.add_argument("--time", type=float, default=8.0)
    ap.add_argument("--exec-seed", type=int, default=42)
    ap.add_argument("--snapshot", default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--out", default="results/ablation/bifurcation.json")
    args = ap.parse_args()

    rst = load_rst()
    snap = rst.load_snapshot(Path(args.snapshot))
    pool = rst.load_pool(Path(args.snapshot).parent / "refined_pool.json")
    learned = snap["pheromones"]["operator_level"]

    print(f"perturbations/eps={args.n_perturb} time={args.time}s exec_seed={args.exec_seed}\n")
    rows = []
    for name in args.instances:
        dm, opt = rst.load_tsplib(f"data/instances/tsp/{name}.tsp")
        opt = opt or OPT.get(name); n = len(dm)
        base = run_one(rst, dm, snap, pool, n, opt, learned, args.exec_seed, args.time)
        print(f"{name} (baseline gap {base:.2f}%):")
        for eps in args.eps:
            if eps == 0.0:
                gaps = [base]
            else:
                gaps = []
                for p in range(args.n_perturb):
                    rng = random.Random(1000 + p)
                    pher = perturb(learned, eps, rng)
                    gaps.append(run_one(rst, dm, snap, pool, n, opt, pher, args.exec_seed, args.time))
            dev = [abs(g - base) for g in gaps]
            rows.append({"instance": name, "eps": eps, "gaps": gaps,
                         "mean_gap": st.mean(gaps), "std_gap": st.pstdev(gaps),
                         "mean_abs_dev": st.mean(dev), "max_abs_dev": max(dev)})
            print(f"  eps={eps:<5} mean_gap={st.mean(gaps):6.2f}%  spread(std)={st.pstdev(gaps):5.2f}  "
                  f"mean|Δ from base|={st.mean(dev):5.2f}pp  max|Δ|={max(dev):5.2f}pp")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
