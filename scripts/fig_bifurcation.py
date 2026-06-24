#!/usr/bin/env python3
"""Bifurcation / perturbation-sensitivity figure (R4.10). Output deviation of the
Symbolic Executor as the learned pheromones are perturbed by magnitude eps. A
smooth, bounded response (no discontinuous jumps) means the learned policy is
stable, consistent with the spectral gap. Reads results/ablation/bifurcation.json.
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt

from paper_palette import PALETTE, apply_paper_style


def main():
    apply_paper_style()
    d = json.load(open("results/ablation/bifurcation.json"))
    by = defaultdict(list)
    for r in d:
        by[r["instance"]].append((r["eps"], r["mean_abs_dev"], r["max_abs_dev"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    colors = [PALETTE["geakg"], PALETTE["ils"]]
    gmax = 0.0
    for (inst, pts), c in zip(by.items(), colors):
        pts.sort()
        eps = [p[0] for p in pts]; mad = [p[1] for p in pts]; mx = [p[2] for p in pts]
        gmax = max(gmax, max(mx))
        ax.plot(eps, mad, "-o", color=c, lw=1.8, ms=5, label=inst, zorder=3)
        ax.fill_between(eps, mad, mx, color=c, alpha=0.12, zorder=1)
    ax.set_xlabel(r"Pheromone perturbation magnitude $\varepsilon$")
    ax.set_ylabel("Output deviation from baseline (pp)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", title="TSP instance")
    ax.set_ylim(0, gmax * 1.12)  # headroom so the max-deviation band does not touch the frame
    fig.tight_layout()
    out = "paper_els_cas/fig_bifurcation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
