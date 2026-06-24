#!/usr/bin/env python3
"""QAP transfer ablation (mirror of the JSSP E1 ablation) -- now with the _qap
pheromone key bug FIXED.

For each (QAP instance, seed) under an identical setup (same transferred operators,
same rules, same Gilmore--Lawler initial solution, same per-seed RNG), three methods
run and the ONLY difference between LEARNED and UNIFORM is the operator pheromones:

  LEARNED : snapshot.pheromones.operator_level remapped to the _qap operator ids
            (what ACO learned, now actually applied)
  UNIFORM : all operator pheromones = 1.0 (learned ordering removed)
  ILS     : from-scratch Iterated Local Search baseline (seeded)

LEARNED vs UNIFORM isolates whether the learned graph drives QAP transfer; LEARNED
vs ILS is the transfer-vs-baseline story. All runs are zero-token (no LLM).

  uv run python scripts/run_qap_ablation_multiseed.py \
      --instances nug12 nug15 nug20 nug25 nug30 chr12a chr15a chr20a tai20a \
      --seeds 5 --time 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from run_qap_transfer import (
    ils_qap, create_qap_evaluate_fn, qap_copy, extract_symbolic_rules,
    SymbolicExecutor, QAPAssignment, QAPDomain, TransferManager,
    QAP_INSTANCE_DIR, QAP_INSTANCES,
)


def symbolic_run(domain, qap_instance, adapted_operators, rule_engine, pheromones,
                 success_freq, gl_assignment, gl_cost, t, seed):
    random.seed(seed)
    ex = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_qap_evaluate_fn(domain),
        copy_fn=qap_copy,
        operator_pheromones=pheromones,
        success_frequency=success_freq,
        global_mode=True,
        verbose=False,
    )
    init = QAPAssignment(assignment=list(gl_assignment), cost=gl_cost)
    res = ex.execute(operators=adapted_operators, initial_solution=init,
                     initial_cost=gl_cost, time_limit=t, instance=qap_instance)
    return float(res.best_solution.cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+",
                    default=["nug12", "nug15", "nug20", "nug25", "nug30",
                             "chr12a", "chr15a", "chr20a", "tai20a"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--out", default="results/case_study_2_optimization/E2_qap_ablation_FIXED.json")
    args = ap.parse_args()

    snap_path = Path(args.snapshot)
    snapshot = json.load(open(snap_path))
    # _qap key-match fix: remap snapshot operator_level keys to the suffix the
    # adapted operators carry, else the learned pheromones are never applied.
    _raw = snapshot["pheromones"]["operator_level"]
    learned = {f"{k.split(chr(58),1)[0]}:{k.split(chr(58),1)[1]}_qap": v for k, v in _raw.items()}
    uniform = {k: 1.0 for k in learned}
    success_freq = {}
    for p in snapshot.get("successful_paths", []):
        for op in p.get("operators", []):
            success_freq[op] = success_freq.get(op, 0) + 1
    rule_engine = extract_symbolic_rules(snapshot)

    domain = QAPDomain()
    manager = TransferManager()
    rows = []
    print(f"QAP ablation | seeds={args.seeds} time={args.time}s | LEARNED vs UNIFORM vs ILS "
          f"(_qap key bug FIXED)\n")
    for name in args.instances:
        path = QAP_INSTANCE_DIR / f"{name}.dat"
        if not path.exists():
            print(f"{name:8} SKIP (not found: {path})"); continue
        inst = domain.load_instance(path)
        opt = QAP_INSTANCES.get(f"{name}.dat", {}).get("optimal")
        res = manager.transfer_from_akg(source_snapshot=str(snap_path),
                                        target_domain="qap", target_instance=inst)
        ops = res.adapted_operators
        gl = domain.gilmore_lawler_solution(inst)
        gl_assign, gl_cost = list(gl.assignment), gl.cost
        for seed in range(args.seeds):
            try:
                ml = symbolic_run(domain, inst, ops, rule_engine, learned,
                                  success_freq, gl_assign, gl_cost, args.time, seed)
                mu = symbolic_run(domain, inst, ops, rule_engine, uniform,
                                  success_freq, gl_assign, gl_cost, args.time, seed)
                _, mi, _ = ils_qap(inst, domain, args.time, seed=seed)
                rows.append({"instance": name, "n": inst.n, "optimal": opt, "seed": seed,
                             "learned": ml, "uniform": mu, "ils": float(mi), "gl": float(gl_cost)})
            except Exception as e:
                rows.append({"instance": name, "seed": seed, "error": str(e)[:140]})
        ok = [r for r in rows if r["instance"] == name and "learned" in r]
        if ok:
            L = [r["learned"] for r in ok]; U = [r["uniform"] for r in ok]; I = [r["ils"] for r in ok]
            def g(x):
                return f"{100*(st.mean(x)-opt)/opt:6.2f}%" if opt else f"{st.mean(x):10.1f}"
            print(f"{name:8} n={inst.n:3} | learned {g(L)} | uniform {g(U)} | ils {g(I)}"
                  f"  | L<U {sum(l<u for l,u in zip(L,U))}/{len(L)}  L<=I {sum(l<=i for l,i in zip(L,I))}/{len(L)}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
