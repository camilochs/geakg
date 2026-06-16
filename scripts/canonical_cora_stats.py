#!/usr/bin/env python3
"""Compute ALL NAS-Bench-Graph Cora analyses from ONE canonical run (run[0] of the
clean 204019 snapshot), so every Cora number in the paper (pheromone convergence
§sec:pheromone_convergence, rule table, spectral, dominant paths/Fig 12) is
sourced from the same run and mutually consistent.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

SRC = "artifacts/nas/nasbench_graph_cora_canonical.json"


def main():
    run = json.load(open(SRC))["runs"][0]
    pher = run["pheromones_display"]              # {"r_i->r_j": tau}
    top_paths = run["top_paths"]
    taus = np.array(list(pher.values()))
    roles = sorted({r for e in pher for r in e.split("->")})

    n_edges = len(pher)
    sat = int((taus >= 0.99).sum())               # saturated at tau_max
    pruned = int((taus <= 0.10).sum())            # near tau_min
    print("=== Pheromone matrix (canonical Cora run) ===")
    print(f"roles: {len(roles)}   edges: {n_edges}")
    print(f"tau range: [{taus.min():.2f}, {taus.max():.2f}]   mean: {taus.mean():.2f}")
    print(f"saturated (tau>=0.99): {sat} ({100*sat/n_edges:.0f}%)   "
          f"pruned (tau<=0.10): {pruned} ({100*pruned/n_edges:.0f}%)")

    # entropy reduction vs uniform (over outgoing distribution per antecedent)
    out = defaultdict(list)
    for e, t in pher.items():
        out[e.split("->")[0]].append(t)
    red = []
    for ri, ts in out.items():
        if len(ts) > 1:
            p = np.array(ts); p = p / p.sum()
            h = -(p * np.log(p)).sum() / np.log(len(p))   # normalized [0,1]
            red.append(1 - h)                              # reduction from uniform
    print(f"mean per-antecedent entropy reduction vs uniform: {100*np.mean(red):.0f}%")

    print("\n=== Top rules by pheromone (for Table 1829 update) ===")
    top = sorted(pher.items(), key=lambda kv: -kv[1])[:7]
    out_mass = defaultdict(float)
    for e, t in pher.items():
        out_mass[e.split("->")[0]] += t
    sup = defaultdict(int)
    for tp in top_paths:
        for a, b in zip(tp["path"], tp["path"][1:]):
            sup[f"{a}->{b}"] += tp.get("count", 1)
    for e, t in top:
        a, b = e.split("->")
        conf = t / out_mass[a]
        print(f"  {a:18}->{b:18} tau={t:.2f} conf={conf:.2f} support={sup.get(e,0)}")

    print("\n=== Dominant paths (for Figure 12) ===")
    for i, tp in enumerate(sorted(top_paths, key=lambda x: -x.get("count", 0))[:5], 1):
        print(f"  #{i} (count={tp.get('count')}): {' -> '.join(tp['path'])}")


if __name__ == "__main__":
    main()
