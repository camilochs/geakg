#!/usr/bin/env python3
"""L1 robustness / operator-quality ablation figure (R1.5; isolates L1).

Holding L0 (roles, topology) and L2 (learned pheromones, rules) fixed, a fraction p
of the L1 operator bodies is degraded in two ways:
  dead  : operator becomes a no-op (returns the solution unchanged) -> isolates
          whether L1 operator CONTENT is load-bearing.
  noisy : operator becomes a random 2-swap (valid but role-agnostic) -> robustness
          to low-quality / adversarial operators.
Mean TSP gap (across instances) vs p, log scale because the dead-pool collapse spans
two orders of magnitude. A steep dead curve and a gentle noisy curve show L1 is
load-bearing yet the engine degrades gracefully under noisy operators.

Reads results/ablation/l1_robustness.json.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from paper_palette import PALETTE, apply_paper_style


def main():
    apply_paper_style()
    d = json.load(open("results/ablation/l1_robustness.json"))
    rows = d["rows"]

    # aggregate across instances: per (mode, frac) -> list of per-instance gap means
    agg = defaultdict(list)
    fracs = set()
    for r in rows:
        agg[(r["mode"], r["frac"])].append(r["gap_mean"])
        fracs.add(r["frac"])
    fracs = sorted(fracs)

    intact = float(np.mean(agg[("intact", 0.0)]))

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for mode, color, label, marker in [
        ("dead", PALETTE["geakg"], "Dead operators (no-op)", "o"),
        ("noisy", PALETTE["ils"], "Noisy operators (random swap)", "s"),
    ]:
        xs = [0.0] + [f for f in fracs if f > 0.0]
        ys, es = [], []
        for f in xs:
            key = ("intact", 0.0) if f == 0.0 else (mode, f)
            vals = agg[key]
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        ax.errorbar(xs, ys, yerr=es, marker=marker, color=color, lw=1.8, ms=6,
                    capsize=3, label=label, zorder=3,
                    markeredgecolor=PALETTE["ink"], markeredgewidth=0.6)

    ax.axhline(intact, color=PALETTE["ink"], lw=0.9, ls="--", alpha=0.7, zorder=1)
    ax.annotate(f"intact pool ({intact:.1f}%)", (0.02, intact), fontsize=8,
                color=PALETTE["ink"], va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of L1 operators degraded ($p$)")
    ax.set_ylabel("Mean TSP gap to optimum (%, log scale)")
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = "paper_els_cas/fig_l1_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")

    # print the headline numbers for the manuscript
    def at(mode, f):
        return float(np.mean(agg[(mode, f)])) if agg[(mode, f)] else float("nan")
    print(f"saved {out}")
    print(f"intact p=0 mean gap = {intact:.2f}%")
    for f in [0.25, 0.5, 0.75, 1.0]:
        print(f"  p={f}: dead {at('dead', f):8.2f}%   noisy {at('noisy', f):7.2f}%")


if __name__ == "__main__":
    main()
