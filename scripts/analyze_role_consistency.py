#!/usr/bin/env python3
"""R1.4 — semantic-role consistency: do operators bound to the SAME role behave
consistently, or is the role label semantically ambiguous?

For each role with >=2 operators, we apply every operator in the role to a fixed
set of random solution states and record the cost change it induces. A role is
semantically coherent if its operators (a) agree on the sign of the change on most
states (same procedural intent) and (b) have comparable effect magnitudes. We
report, per role: #operators, mean |delta|, direction-agreement (fraction of states
where all operators in the role move the cost the same way), and the cross-operator
spread of mean effect (coefficient of variation). Post-hoc on the frozen pool; no
tokens.

  uv run python scripts/analyze_role_consistency.py
"""

from __future__ import annotations

import importlib.util
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_rst():
    spec = importlib.util.spec_from_file_location("rst", "scripts/run_symbolic_tsp.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    rst = load_rst()
    snap_dir = Path("experiments/iterative/20260125_121313_iterative")
    pool = rst.load_pool(snap_dir / "refined_pool.json")
    dm, _ = rst.load_tsplib("data/instances/tsp/berlin52.tsp")
    ctx = rst.TSPContext(dm)
    n = len(dm)
    ops = rst.compile_operators(pool, ctx)

    by_role = defaultdict(list)
    for op in ops:
        by_role[op.role].append(op)

    random.seed(0)
    states = []
    for _ in range(40):                       # fixed shared set of states
        t = list(range(n)); random.shuffle(t)
        states.append(t)

    print(f"berlin52  n={n}  states={len(states)}  total operators={len(ops)}\n")
    print(f"{'role':24}{'#ops':>5}{'mean|Δ|':>10}{'dir-agree':>11}{'effect-CV':>11}  coherent?")
    print("-" * 76)
    rows = []
    for role, rops in sorted(by_role.items()):
        if len(rops) < 2:
            continue
        # per-operator effect per state
        eff = {op.name: [] for op in rops}
        for s in states:
            base = ctx.evaluate(s)
            for op in rops:
                try:
                    ns = op.adapted_fn(list(s))
                    eff[op.name].append(ctx.evaluate(ns) - base)
                except Exception:
                    eff[op.name].append(0.0)
        # direction agreement: fraction of states where all operators share sign
        agree = 0
        for i in range(len(states)):
            signs = {(-1 if eff[op.name][i] < -1e-9 else (1 if eff[op.name][i] > 1e-9 else 0))
                     for op in rops}
            if len(signs) == 1:
                agree += 1
        dir_agree = agree / len(states)
        # cross-operator spread of mean effect
        op_means = [st.mean(eff[op.name]) for op in rops]
        mu = st.mean(op_means)
        cv = (st.pstdev(op_means) / abs(mu)) if abs(mu) > 1e-9 else float("nan")
        mean_abs = st.mean([abs(x) for v in eff.values() for x in v])
        coherent = "yes" if dir_agree >= 0.7 else ("partial" if dir_agree >= 0.5 else "AMBIGUOUS")
        cv_s = f"{cv:8.2f}" if cv == cv else "     n/a"
        print(f"{role:24}{len(rops):>5}{mean_abs:>10.0f}{dir_agree:>11.2f}{cv_s:>11}  {coherent}")
        rows.append((role, len(rops), dir_agree, cv))

    multi = [r for r in rows]
    print(f"\nroles with >=2 operators: {len(multi)}")
    print(f"mean direction-agreement across multi-operator roles: "
          f"{st.mean([r[2] for r in multi]):.2f}")
    print(f"roles semantically coherent (dir-agree>=0.7): "
          f"{sum(1 for r in multi if r[2] >= 0.7)}/{len(multi)}")


if __name__ == "__main__":
    main()
