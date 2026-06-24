#!/usr/bin/env python3
"""L0 ontology ablation figure: does the RoleSchema ontology constraint help, or
just limit diversity (R1.1)? Compares the REAL DYNAMIC offline ACO trained on a
CONSTRAINED topology (only valid category transitions) vs an UNRESTRICTED topology
(every role->role edge). Lower gap is better; constrained below unrestricted means
the ontology constraint improves search, not merely restricts it.

Reads results/ablation/L0_dynamic_<instance>.json (multi-seed).
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paper_palette import PALETTE, apply_paper_style

INSTANCES = ["berlin52", "kroA100", "ch150"]


def main():
    apply_paper_style()
    labels, Cm, Cs, Um, Us = [], [], [], [], []
    for name in INSTANCES:
        p = Path(f"results/ablation/L0_dynamic_{name}.json")
        if not p.exists():
            continue
        d = json.load(open(p))
        C = [r["constrained"] for r in d]; U = [r["unrestricted"] for r in d]
        labels.append(name)
        Cm.append(st.mean(C)); Cs.append(st.pstdev(C))
        Um.append(st.mean(U)); Us.append(st.pstdev(U))
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(x - w / 2, Cm, w, yerr=Cs, capsize=3, color=PALETTE["geakg"], edgecolor=PALETTE["ink"],
           lw=0.5, label="Constrained (ontology: valid transitions)",
           error_kw=dict(lw=0.8, ecolor=PALETTE["ink"]))
    ax.bar(x + w / 2, Um, w, yerr=Us, capsize=3, color=PALETTE["classic"], edgecolor=PALETTE["ink"],
           lw=0.5, label="Unrestricted (all role pairs)",
           error_kw=dict(lw=0.8, ecolor=PALETTE["ink"]))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("TSP gap to optimum (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = "paper_els_cas/fig_l0_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    nC = sum(1 for c, u in zip(Cm, Um) if c < u)
    print(f"saved {out} | constrained (ontology) better on {nC}/{len(labels)} instances (mean)")


if __name__ == "__main__":
    main()
