#!/usr/bin/env python3
"""E1 — decisive multi-seed JSSP transfer ablation.

Per (instance, seed), under an identical setup (same transferred operators, same
rules/phases, same initial baseline solution, same per-seed RNG), we run three
methods and the ONLY difference between LEARNED and UNIFORM is the operator
pheromones:

  LEARNED : snapshot.pheromones.operator_level   (what ACO learned)
  UNIFORM : all operator pheromones = 1.0         (learned ordering removed)
  ILS     : from-scratch Iterated Local Search baseline (seeded)

LEARNED vs UNIFORM isolates whether the *learned graph* drives transfer (the
control the reviewers demanded). LEARNED vs ILS is the transfer-vs-baseline story.
All runs are zero-token (no LLM). Reports mean/SD per instance and saves raw JSON.

Usage:
  uv run python scripts/run_jssp_ablation_multiseed.py \
      --instances ft06 ft10 ... --seeds 10 --time 30 \
      --snapshot experiments/iterative/20260125_121313_iterative/akg_snapshot.json
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


def symbolic_run(domain, inst, snapshot, adapted_operators, rule_engine,
                 pheromones, success_freq, baseline_sol, baseline_mk, t, seed):
    random.seed(seed)
    ex = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_jssp_evaluate_fn(domain),
        copy_fn=jssp_copy,
        operator_pheromones=pheromones,
        success_frequency=success_freq,
        global_mode=True,
        verbose=False,
    )
    init = JSSPSchedule(schedule=list(baseline_sol.schedule), makespan=int(baseline_mk))
    res = ex.execute(operators=adapted_operators, initial_solution=init,
                     initial_cost=baseline_mk, time_limit=t, instance=inst)
    return float(res.best_solution.makespan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--out", default="results/case_study_2_optimization/E1_jssp_ablation_multiseed.json")
    args = ap.parse_args()

    snap_path = Path(args.snapshot)
    snapshot = json.load(open(snap_path))
    # FIX (key-match bug): JSSP-adapted operators carry a _jssp suffix, so the
    # snapshot operator_level keys must be remapped to match, else the learned
    # pheromones are never applied (both arms silently fall back to base_weight).
    _raw = snapshot["pheromones"]["operator_level"]
    learned = {f"{k.split(chr(58),1)[0]}:{k.split(chr(58),1)[1]}_jssp": v for k, v in _raw.items()}
    uniform = {k: 1.0 for k in learned}
    success_freq = {}
    for p in snapshot.get("successful_paths", []):
        for op in p.get("operators", []):
            success_freq[op] = success_freq.get(op, 0) + 1
    rule_engine = extract_symbolic_rules(snapshot)

    domain = JSSPDomain()
    manager = TransferManager()
    rows = []
    print(f"E1 ablation | seeds={args.seeds} time={args.time}s | LEARNED vs UNIFORM vs ILS\n")
    for name in args.instances:
        inst = domain.load_instance(Path(f"data/instances/jssp/parsed/{name}.txt"))
        opt = inst.optimal_makespan
        res = manager.transfer_from_akg(source_snapshot=str(snap_path),
                                        target_domain="jssp", target_instance=inst)
        ops = res.adapted_operators
        spt = spt_dispatch(inst); spt_mk = domain.evaluate_solution(spt, inst)
        lpt = lpt_dispatch(inst); lpt_mk = domain.evaluate_solution(lpt, inst)
        base_sol, base_mk = (spt, spt_mk) if spt_mk <= lpt_mk else (lpt, lpt_mk)
        for seed in range(args.seeds):
            try:
                ml = symbolic_run(domain, inst, snapshot, ops, rule_engine, learned,
                                  success_freq, base_sol, base_mk, args.time, seed)
                mu = symbolic_run(domain, inst, snapshot, ops, rule_engine, uniform,
                                  success_freq, base_sol, base_mk, args.time, seed)
                _, mi, _ = ils_jssp(inst, domain, args.time, seed=seed)
                rows.append({"instance": name, "ops": inst.dimension, "optimal": opt,
                             "seed": seed, "learned": ml, "uniform": mu, "ils": float(mi)})
            except Exception as e:
                rows.append({"instance": name, "seed": seed, "error": str(e)[:120]})
        ok = [r for r in rows if r["instance"] == name and "learned" in r]
        if ok:
            L = [r["learned"] for r in ok]; U = [r["uniform"] for r in ok]; I = [r["ils"] for r in ok]
            print(f"{name:6} ops={inst.dimension:3} | learned {st.mean(L):8.1f}±{(st.pstdev(L)):5.1f} "
                  f"| uniform {st.mean(U):8.1f}±{st.pstdev(U):5.1f} | ils {st.mean(I):8.1f}±{st.pstdev(I):5.1f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
