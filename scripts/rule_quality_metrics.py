#!/usr/bin/env python3
"""Rule-quality metrics for the learned symbolic rules (R4.8: confidence, support,
redundancy elimination), computed from the canonical NAS-Bench-Graph Cora run.

For each candidate transition rule r_i -> r_j (antecedent -> consequent):
  confidence = P(r_j | r_i) = tau_ij / sum_k tau_ik   (normalized pheromone)
  support    = number of successful-trace occurrences of r_i immediately
               followed by r_j, summed over the top execution paths weighted by
               their counts (AMIE-style discrete support from traces)
Redundancy elimination: keep only rules whose confidence exceeds a threshold and
that are the dominant consequent for their antecedent; report how many of the 42
candidate transitions survive as non-redundant rules.
"""

from __future__ import annotations

import json
from collections import defaultdict

SRC = "artifacts/nas/nasbench_graph_cora_canonical.json"
CONF_THRESHOLD = 0.5


def main():
    run = json.load(open(SRC))["runs"][0]
    pher = run["pheromones_display"]          # {"r_i->r_j": tau}
    top_paths = run["top_paths"]              # [{"path": [...], "count": n}]

    # outgoing pheromone mass per antecedent -> normalized confidence
    out_mass = defaultdict(float)
    for edge, tau in pher.items():
        ri = edge.split("->")[0]
        out_mass[ri] += tau

    # support: count r_i->r_j occurrences across top paths (weighted by path count)
    support = defaultdict(int)
    for tp in top_paths:
        path, c = tp["path"], tp.get("count", 1)
        for a, b in zip(path, path[1:]):
            support[f"{a}->{b}"] += c

    rules = []
    for edge, tau in pher.items():
        ri, rj = edge.split("->")
        conf = tau / out_mass[ri] if out_mass[ri] > 0 else 0.0
        rules.append({"ant": ri, "cons": rj, "tau": tau, "conf": conf,
                      "support": support.get(edge, 0)})

    # dominant consequent per antecedent (for redundancy)
    best_per_ant = {}
    for r in rules:
        if r["ant"] not in best_per_ant or r["conf"] > best_per_ant[r["ant"]]["conf"]:
            best_per_ant[r["ant"]] = r
    non_redundant = [r for r in rules
                     if r["conf"] >= CONF_THRESHOLD and best_per_ant[r["ant"]] is r]

    print(f"candidate transitions: {len(rules)}")
    print(f"confidence range: {min(r['conf'] for r in rules):.2f}--{max(r['conf'] for r in rules):.2f}")
    print(f"non-redundant rules (conf>={CONF_THRESHOLD}, dominant per antecedent): "
          f"{len(non_redundant)}")
    print(f"rules with trace support>0: {sum(1 for r in rules if r['support']>0)}")
    print()
    print(f"{'antecedent':18} {'consequent':18} {'conf':>5} {'tau':>5} {'support':>7}")
    for r in sorted(non_redundant, key=lambda x: -x["conf"]):
        print(f"{r['ant']:18} {r['cons']:18} {r['conf']:>5.2f} {r['tau']:>5.2f} {r['support']:>7}")

    json.dump({"n_candidates": len(rules), "n_non_redundant": len(non_redundant),
               "conf_threshold": CONF_THRESHOLD,
               "rules": sorted(non_redundant, key=lambda x: -x["conf"])},
              open("results/nas_bench_graph/rule_quality_cora.json", "w"), indent=2)


if __name__ == "__main__":
    main()
