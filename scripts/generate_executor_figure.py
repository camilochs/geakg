#!/usr/bin/env python3
"""Regenerate the Symbolic Executor architecture schematic (fig_symbolic_executor)
in the pink paper palette. Online runtime: a frozen GEAKG snapshot is interpreted
by a zero-token loop. Replaces the old static green diagram.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_palette import PALETTE, apply_paper_style


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0.9, edgecolor=PALETTE["ink"], facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold")


def arrow(ax, p0, p1, rad=0.0, color=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 connectionstyle=f"arc3,rad={rad}", lw=1.4,
                                 color=color or PALETTE["ink"], zorder=1))


def main():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_axis_off()

    # Frozen snapshot (the deployable artifact) with three layer pills
    box(ax, 0.4, 5.5, 9.2, 1.2, "", PALETTE["geakg_soft"], tc=PALETTE["ink"])
    ax.text(5.0, 6.45, "Frozen GEAKG snapshot  (zero LLM tokens at deployment)",
            ha="center", va="center", fontsize=8.6, color=PALETTE["ink"], fontweight="bold")
    box(ax, 0.8, 5.62, 2.6, 0.62, "L0 topology", PALETTE["geakg"], fs=7.8)
    box(ax, 3.7, 5.62, 2.6, 0.62, "L1 operators", PALETTE["accent"], fs=7.8)
    box(ax, 6.6, 5.62, 2.8, 0.62, "L2 pheromones + rules", PALETTE["ils"], fs=7.4)

    # Online loop: four stages
    y = 2.6; h = 1.1; w = 2.05
    xs = [0.4, 2.7, 5.0, 7.5]
    labels = [
        "Rule engine\nstate $\\to$ phase\n(refine / explore / restart)",
        "Role selection\npheromone-weighted\nover eligible roles",
        "Operator selection\nand application",
        "Domain binding\nevaluate / validate\n/ decode",
    ]
    cols = [PALETTE["geakg"], PALETTE["ils"], PALETTE["accent"], PALETTE["classic"]]
    centers = []
    for x, lab, c in zip(xs, labels, cols):
        box(ax, x, y, w, h, lab, c, tc="white" if c != PALETTE["classic"] else PALETTE["ink"], fs=7.0)
        centers.append((x + w / 2, y + h / 2))

    # snapshot feeds the loop
    arrow(ax, (5.0, 5.5), (xs[0] + w / 2, y + h), rad=0.0, color=PALETTE["ink"])
    # forward arrows
    for i in range(3):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2), color=PALETTE["ink"])
    # loop back (domain binding -> rule engine) below the row
    ax.add_patch(FancyArrowPatch((centers[3][0], y), (centers[0][0], y),
                 arrowstyle="-|>", mutation_scale=13, connectionstyle="arc3,rad=-0.32",
                 lw=1.4, color=PALETTE["geakg"], zorder=1))
    ax.text(5.0, 0.95, "iterate until budget exhausted", ha="center", va="center",
            fontsize=7.6, color=PALETTE["geakg"], style="italic")

    fig.tight_layout()
    out = "paper_els_cas/fig_symbolic_executor.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
