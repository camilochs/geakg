#!/usr/bin/env python3
"""Budget-scaling fairness test for the JSSP transfer win (both methods UNTUNED).

The large-JSSP advantage of the transferred GEAKG snapshot over a from-scratch ILS
could be an artifact of ILS getting too few iterations in a fixed wall-clock budget.
We test this directly and honestly: hold GEAKG at the base budget and give ILS
PROGRESSIVELY MORE wall-clock (1x, 5x). If GEAKG at 1x still beats ILS at 5x, the
advantage is transferred-knowledge, not iteration starvation. Neither method is
tuned for JSSP (GEAKG is a frozen TSP snapshot; ILS uses default parameters), so
the comparison is symmetric in tuning.

  uv run python scripts/run_jssp_evalparity.py --instances abz7 la31 abz9 --seeds 3 --base 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from run_jssp_transfer import (
    ils_jssp, spt_dispatch, lpt_dispatch, create_jssp_evaluate_fn, jssp_copy,
    extract_symbolic_rules, SymbolicExecutor, JSSPSchedule, JSSPDomain,
)
from src.geakg.transfer.transfer_manager import TransferManager


def geakg_run(domain, inst, ops, rule_engine, pher, success_freq, base_sol, base_mk, t, seed):
    random.seed(seed)
    ex = SymbolicExecutor(rule_engine=rule_engine, evaluate_fn=create_jssp_evaluate_fn(domain),
                          copy_fn=jssp_copy, operator_pheromones=pher, success_frequency=success_freq,
                          global_mode=True, verbose=False)
    init = JSSPSchedule(schedule=list(base_sol.schedule), makespan=int(base_mk))
    r = ex.execute(operators=ops, initial_solution=init, initial_cost=base_mk, time_limit=t, instance=inst)
    return float(r.best_solution.makespan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["abz7", "la31", "abz9"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base", type=float, default=30.0)
    ap.add_argument("--mults", nargs="+", type=float, default=[1.0, 5.0])
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--out", default="results/case_study_2_optimization/E4_jssp_evalparity.json")
    args = ap.parse_args()

    snapshot = json.load(open(args.snapshot))
    _raw = snapshot["pheromones"]["operator_level"]
    learned = {f"{k.split(chr(58),1)[0]}:{k.split(chr(58),1)[1]}_jssp": v for k, v in _raw.items()}
    success_freq = {}
    for p in snapshot.get("successful_paths", []):
        for op in p.get("operators", []):
            success_freq[op] = success_freq.get(op, 0) + 1
    rule_engine = extract_symbolic_rules(snapshot)
    domain = JSSPDomain()
    manager = TransferManager()

    rows = []
    hdr = "  ".join([f"ILS@{m:g}x" for m in args.mults])
    print(f"JSSP budget-scaling | base={args.base}s seeds={args.seeds} | GEAKG@1x vs {hdr} (all untuned)\n")
    for name in args.instances:
        inst = domain.load_instance(Path(f"data/instances/jssp/parsed/{name}.txt"))
        opt = inst.optimal_makespan
        ops = manager.transfer_from_akg(source_snapshot=args.snapshot, target_domain="jssp",
                                        target_instance=inst).adapted_operators
        spt = spt_dispatch(inst); spt_mk = domain.evaluate_solution(spt, inst)
        lpt = lpt_dispatch(inst); lpt_mk = domain.evaluate_solution(lpt, inst)
        base_sol, base_mk = (spt, spt_mk) if spt_mk <= lpt_mk else (lpt, lpt_mk)
        g, ils_by_m = [], {m: [] for m in args.mults}
        iters_by_m = {m: [] for m in args.mults}
        for seed in range(args.seeds):
            g.append(geakg_run(domain, inst, ops, rule_engine, learned, success_freq,
                               base_sol, base_mk, args.base, seed))
            for m in args.mults:
                _, mk, it = ils_jssp(inst, domain, args.base * m, seed=seed)
                ils_by_m[m].append(float(mk)); iters_by_m[m].append(it)
        gm = st.mean(g)
        def gap(x): return f"{100*(st.mean(x)-opt)/opt:5.1f}%" if opt else f"{st.mean(x):.0f}"
        parts = []
        for m in args.mults:
            beat = "GEAKG" if gm < st.mean(ils_by_m[m]) else "ILS"
            parts.append(f"ILS@{m:g}x {gap(ils_by_m[m])}({st.mean(iters_by_m[m]):.0f}it)->{beat}")
        print(f"{name:6} opt={opt} | GEAKG@1x {gap(g)} | " + " | ".join(parts))
        rows.append({"instance": name, "opt": opt, "geakg_1x": g,
                     "ils": {str(m): ils_by_m[m] for m in args.mults},
                     "ils_iters": {str(m): iters_by_m[m] for m in args.mults}})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
