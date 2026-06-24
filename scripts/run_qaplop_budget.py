#!/usr/bin/env python3
"""Loss-side budget-invariance control for the when-pays boundary (QAP + LOP).

The when-pays characterization operationalizes "expensive target-side search" as a
gap that does NOT close under more budget. On the WIN side (large JSSP) the
eval-parity check showed GEAKG beats ILS even when ILS gets 5x the budget. This is
the symmetric control on the LOSS side: on QAP and LOP, where GEAKG loses, we give
ILS only ONE-FIFTH of the budget and ask whether it STILL beats GEAKG at full
budget. If it does, the target-side local search is genuinely "cheap" (reaches
near-optimal with little compute), so the boundary is a measured property, not a
case where GEAKG happened to lose under-budget.

  uv run python scripts/run_qaplop_budget.py --seeds 3 --time 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from run_qap_ablation_multiseed import symbolic_run as qap_sym
from run_qap_transfer import ils_qap, extract_symbolic_rules, QAPDomain, QAP_INSTANCES, QAP_INSTANCE_DIR
from run_lop_ablation_multiseed import symbolic_run as lop_sym, LOP_DIR
from run_lop_transfer import ils_lop, LOPDomain
from src.geakg.transfer.transfer_manager import TransferManager

SNAP = "experiments/iterative/20260125_121313_iterative/akg_snapshot.json"


def setup(suffix):
    snap = json.load(open(SNAP))
    raw = snap["pheromones"]["operator_level"]
    learned = {f"{k.split(chr(58),1)[0]}:{k.split(chr(58),1)[1]}_{suffix}": v for k, v in raw.items()}
    sf = {}
    for p in snap.get("successful_paths", []):
        for op in p.get("operators", []):
            sf[op] = sf.get(op, 0) + 1
    return extract_symbolic_rules(snap), learned, sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--fifth", type=float, default=0.2)
    ap.add_argument("--qap", nargs="+", default=["nug30", "tai50a", "sko100a"])
    ap.add_argument("--lop", nargs="+", default=["be75np", "stabu70", "t65b11xx"])
    ap.add_argument("--out", default="results/case_study_2_optimization/E6_qaplop_budget.json")
    args = ap.parse_args()
    T, Tf = args.time, args.time * args.fifth
    mgr = TransferManager()
    rows = []
    print(f"Loss-side budget control | GEAKG@{T:g}s vs ILS@{T:g}s vs ILS@{Tf:g}s (1/5) | seeds={args.seeds}\n")

    # --- QAP ---
    re_q, learned_q, sf_q = setup("qap")
    dom = QAPDomain()
    for name in args.qap:
        inst = dom.load_instance(QAP_INSTANCE_DIR / f"{name}.dat")
        opt = QAP_INSTANCES.get(f"{name}.dat", {}).get("optimal")
        ops = mgr.transfer_from_akg(source_snapshot=SNAP, target_domain="qap", target_instance=inst).adapted_operators
        gl = dom.gilmore_lawler_solution(inst)
        G, I1, If = [], [], []
        for s in range(args.seeds):
            G.append(qap_sym(dom, inst, ops, re_q, learned_q, sf_q, list(gl.assignment), gl.cost, T, s))
            _, i1, _ = ils_qap(inst, dom, T, seed=s); I1.append(float(i1))
            _, iv, _ = ils_qap(inst, dom, Tf, seed=s); If.append(float(iv))
        g = lambda x: f"{100*(st.mean(x)-opt)/opt:5.2f}%" if opt else f"{st.mean(x):.0f}"
        verdict = "ILS@1/5 STILL beats GEAKG" if st.mean(If) < st.mean(G) else "GEAKG catches up at ILS@1/5"
        print(f"QAP {name:8} | GEAKG@1x {g(G)} | ILS@1x {g(I1)} | ILS@1/5 {g(If)}  -> {verdict}")
        rows.append({"domain": "qap", "instance": name, "opt": opt, "geakg": G, "ils_1x": I1, "ils_fifth": If})

    # --- LOP (maximization: higher value better) ---
    re_l, learned_l, sf_l = setup("lop")
    dl = LOPDomain()
    for name in args.lop:
        inst = dl.load_instance(LOP_DIR / f"{name}.txt")
        ops = mgr.transfer_from_akg(source_snapshot=SNAP, target_domain="lop", target_instance=inst).adapted_operators
        becker = dl.becker_solution(inst)
        G, I1, If = [], [], []
        for s in range(args.seeds):
            G.append(lop_sym(dl, inst, ops, re_l, learned_l, sf_l, becker.permutation, becker.value, T, s))
            _, i1, _ = ils_lop(inst, dl, T, seed=s); I1.append(float(i1))
            _, iv, _ = ils_lop(inst, dl, Tf, seed=s); If.append(float(iv))
        verdict = "ILS@1/5 STILL beats GEAKG" if st.mean(If) > st.mean(G) else "GEAKG catches up at ILS@1/5"
        print(f"LOP {name:8} | GEAKG@1x {st.mean(G):10.0f} | ILS@1x {st.mean(I1):10.0f} | ILS@1/5 {st.mean(If):10.0f}  -> {verdict}")
        rows.append({"domain": "lop", "instance": name, "geakg": G, "ils_1x": I1, "ils_fifth": If})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
