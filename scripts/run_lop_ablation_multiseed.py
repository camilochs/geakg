#!/usr/bin/env python3
"""LOP transfer ablation (mirror of the JSSP/QAP ablations) -- _lop key bug FIXED.

LOP (Linear Ordering Problem) is a permutation problem whose landscape is a
SEQUENCING/ordering landscape, structurally closer to TSP than QAP's quadratic
landscape -- so it is the cleaner test of "structure transfers when the landscape
is compatible". Per (instance, seed), identical setup (transferred operators,
rules, Becker initial solution, per-seed RNG); the ONLY difference between LEARNED
and UNIFORM is the operator pheromones:

  LEARNED : snapshot.pheromones.operator_level remapped to the _lop operator ids
  UNIFORM : all operator pheromones = 1.0
  ILS     : from-scratch swap-based Iterated Local Search (seeded), untuned

LOP is a MAXIMIZATION problem; we report gap-to-optimum = (opt - value)/opt (%),
lower is better. All runs are zero-token.

  uv run python scripts/run_lop_ablation_multiseed.py \
      --instances be75eec be75np be75oi be75tot stabu70 stabu74 t65b11xx t65i11xx \
      --seeds 5 --time 15
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from run_lop_transfer import (
    ils_lop, create_lop_evaluate_fn, lop_copy, LOPPermutation,
    extract_symbolic_rules, SymbolicExecutor,
)
from src.domains.lop import LOPDomain
from src.geakg.transfer.transfer_manager import TransferManager

LOP_DIR = Path("experiments/nsse-transfer/instances/lop")


def symbolic_run(domain, inst, ops, rule_engine, pher, success_freq, becker_perm, becker_val, t, seed):
    random.seed(seed)
    ex = SymbolicExecutor(rule_engine=rule_engine, evaluate_fn=create_lop_evaluate_fn(domain),
                          copy_fn=lop_copy, operator_pheromones=pher, success_frequency=success_freq,
                          global_mode=True, verbose=False)
    init = LOPPermutation(permutation=list(becker_perm), value=becker_val)
    res = ex.execute(operators=ops, initial_solution=init, initial_cost=-becker_val,
                     time_limit=t, instance=inst)
    return -float(res.best_cost)            # LOP value (maximize) = -minimized cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+",
                    default=["be75eec", "be75np", "be75oi", "be75tot",
                             "stabu70", "stabu74", "t65b11xx", "t65i11xx"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time", type=float, default=15.0)
    ap.add_argument("--snapshot",
                    default="experiments/iterative/20260125_121313_iterative/akg_snapshot.json")
    ap.add_argument("--out", default="results/case_study_2_optimization/E5_lop_ablation.json")
    args = ap.parse_args()

    snapshot = json.load(open(args.snapshot))
    _raw = snapshot["pheromones"]["operator_level"]
    learned = {f"{k.split(chr(58),1)[0]}:{k.split(chr(58),1)[1]}_lop": v for k, v in _raw.items()}
    uniform = {k: 1.0 for k in learned}
    success_freq = {}
    for p in snapshot.get("successful_paths", []):
        for op in p.get("operators", []):
            success_freq[op] = success_freq.get(op, 0) + 1
    rule_engine = extract_symbolic_rules(snapshot)

    domain = LOPDomain()
    manager = TransferManager()
    rows = []
    print(f"LOP ablation | seeds={args.seeds} time={args.time}s | LEARNED vs UNIFORM vs ILS "
          f"(_lop key bug FIXED; LOP maximizes, gap=(opt-val)/opt)\n")
    for name in args.instances:
        path = LOP_DIR / f"{name}.txt"
        if not path.exists():
            print(f"{name:9} SKIP (not found)"); continue
        inst = domain.load_instance(path)
        opt = inst.optimal_value
        ops = manager.transfer_from_akg(source_snapshot=args.snapshot, target_domain="lop",
                                        target_instance=inst).adapted_operators
        becker = domain.becker_solution(inst)
        for seed in range(args.seeds):
            try:
                vl = symbolic_run(domain, inst, ops, rule_engine, learned, success_freq,
                                  becker.permutation, becker.value, args.time, seed)
                vu = symbolic_run(domain, inst, ops, rule_engine, uniform, success_freq,
                                  becker.permutation, becker.value, args.time, seed)
                _, vi, _ = ils_lop(inst, domain, args.time, seed=seed)
                rows.append({"instance": name, "n": inst.n, "optimal": opt, "seed": seed,
                             "learned": vl, "uniform": vu, "ils": float(vi)})
            except Exception as e:
                rows.append({"instance": name, "seed": seed, "error": str(e)[:140]})
        ok = [r for r in rows if r["instance"] == name and "learned" in r]
        if ok:
            L = [r["learned"] for r in ok]; U = [r["uniform"] for r in ok]; I = [r["ils"] for r in ok]
            # LOP maximizes -> higher value is better. Compare by value (no optimum needed).
            lwin = sum(l > u for l, u in zip(L, U)); giwin = sum(l >= i for l, i in zip(L, I))
            optstr = f" gapL={100*(opt-st.mean(L))/opt:.1f}%" if opt else ""
            print(f"{name:9} n={inst.n:3} | learned {st.mean(L):10.0f} | uniform {st.mean(U):10.0f} "
                  f"| ils {st.mean(I):10.0f}  | L>U {lwin}/{len(L)}  GEAKG>=ILS {giwin}/{len(L)}{optstr}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
