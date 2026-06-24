#!/usr/bin/env python3
"""L2 ablation figure: are the learned ACO pheromones decorative? Compares the
Symbolic Executor with LEARNED operator pheromones vs UNIFORM pheromones (learned
ordering removed), holding operators, rules, phases, initial solution and per-seed
RNG fixed. Lower gap is better; learned below uniform => the learned ordering
(L2) carries real procedural knowledge.

Reads TSP from results/ablation/pheromone_ablation_confirm.json and (if present)
JSSP from results/case_study_2_optimization/E1_jssp_ablation_multiseed.json.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paper_palette import PALETTE, apply_paper_style


def tsp_rows():
    d = json.load(open("results/ablation/pheromone_ablation_confirm.json"))
    out = []
    for r in d["rows"]:
        L = r["learned"]; U = r["uniform"]
        out.append((f"{r['instance']}", st.mean(L), st.pstdev(L), st.mean(U), st.pstdev(U)))
    return out


def jssp_rows():
    p = Path("results/case_study_2_optimization/E1_jssp_ablation_multiseed.json")
    if not p.exists():
        return []
    d = json.load(open(p))
    by = {}
    for r in d:
        if "learned" in r:
            by.setdefault(r["instance"], {"L": [], "U": [], "opt": r.get("optimal")})
            by[r["instance"]]["L"].append(r["learned"])
            by[r["instance"]]["U"].append(r["uniform"])
    out = []
    for inst, v in by.items():
        opt = v["opt"]
        # express JSSP as gap% if optimal known, else relative to uniform mean
        if opt:
            L = [100 * (x - opt) / opt for x in v["L"]]
            U = [100 * (x - opt) / opt for x in v["U"]]
        else:
            base = st.mean(v["U"])
            L = [100 * (x - base) / base for x in v["L"]]
            U = [100 * (x - base) / base for x in v["U"]]
        out.append((inst, st.mean(L), st.pstdev(L), st.mean(U), st.pstdev(U)))
    return out


def main():
    apply_paper_style()
    tsp = tsp_rows()
    jssp = jssp_rows()
    rows = tsp + jssp
    labels = [r[0] for r in rows]
    Lm = np.array([r[1] for r in rows]); Ls = np.array([r[2] for r in rows])
    Um = np.array([r[3] for r in rows]); Us = np.array([r[4] for r in rows])

    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(rows)), 4.0))
    ax.bar(x - w / 2, Lm, w, yerr=Ls, capsize=2.5, color=PALETTE["geakg"],
           edgecolor=PALETTE["ink"], lw=0.5, label="Learned pheromones (L2)",
           error_kw=dict(lw=0.8, ecolor=PALETTE["ink"]))
    ax.bar(x + w / 2, Um, w, yerr=Us, capsize=2.5, color=PALETTE["classic"],
           edgecolor=PALETTE["ink"], lw=0.5, label="Uniform (learned ordering removed)",
           error_kw=dict(lw=0.8, ecolor=PALETTE["ink"]))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Gap to optimum (%)")
    if jssp:
        ax.axvline(len(tsp) - 0.5, color=PALETTE["ink"], lw=0.7, ls=":")
        ax.text(len(tsp) / 2 - 0.5, ax.get_ylim()[1] * 0.96, "TSP", ha="center", fontsize=8, color=PALETTE["ink"])
        ax.text(len(tsp) + len(jssp) / 2 - 0.5, ax.get_ylim()[1] * 0.96, "JSSP (transfer)", ha="center", fontsize=8, color=PALETTE["ink"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = "paper_els_cas/fig_l2_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    nL = sum(1 for r in rows if r[1] < r[3])
    print(f"saved {out}  | learned<=uniform on {nL}/{len(rows)} instances ({len(tsp)} TSP, {len(jssp)} JSSP)")


if __name__ == "__main__":
    main()
