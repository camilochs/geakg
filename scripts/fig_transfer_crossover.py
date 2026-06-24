#!/usr/bin/env python3
"""Transfer crossover figure: GEAKG advantage over a from-scratch ILS baseline
grows with JSSP instance size (single-seed, 60 s/instance, fixed precedence-aware
eval). Headline figure for the size-dependent transfer thesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paper_palette import PALETTE, apply_paper_style

from src.domains.jssp import JSSPDomain

# (instance, GEAKG transfer makespan, ILS makespan) — regenerated single-seed run
DATA = [
    ("ft06", 60, 57), ("ft10", 1348, 1350), ("ft20", 1633, 1416),
    ("la01", 747, 682), ("la06", 1023, 1007), ("la16", 1199, 1286),
    ("la21", 1486, 1917), ("la26", 1881, 2801), ("la31", 2479, 4312),
    ("la36", 1976, 2968), ("la40", 2005, 3043),
    ("abz5", 1485, 1675), ("abz7", 1149, 2210), ("abz9", 1164, 2160),
]


def main():
    apply_paper_style()
    dom = JSSPDomain()
    sizes, adv, names, win = [], [], [], []
    for name, geakg, ils in DATA:
        inst = dom.load_instance(Path(f"data/instances/jssp/parsed/{name}.txt"))
        sizes.append(inst.dimension)                 # n_jobs * n_machines
        adv.append(100.0 * (ils - geakg) / ils)      # +ve => GEAKG better
        names.append(name)
        win.append(geakg < ils)
    sizes, adv = np.array(sizes), np.array(adv)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # shaded regions
    ax.axhspan(0, max(adv) + 8, color=PALETTE["geakg_soft"], alpha=0.30, zorder=0)
    ax.axhline(0, color=PALETTE["ink"], lw=1.0, ls="--", zorder=1)

    # linear trend (advantage grows with size)
    b, a = np.polyfit(sizes, adv, 1)
    xs = np.linspace(sizes.min(), sizes.max(), 100)
    ax.plot(xs, a + b * xs, color=PALETTE["geakg"], lw=1.6, alpha=0.55, zorder=2)
    xcross = -a / b  # size where the trend crosses zero
    ax.axvline(xcross, color=PALETTE["ink"], lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.annotate(f"crossover ≈ {xcross:.0f} ops", (xcross, max(adv) + 2),
                fontsize=8.5, color=PALETTE["ink"], ha="center")

    # points: pink where GEAKG wins, periwinkle where ILS wins.
    # Per-instance label offsets (points) to de-collide overlapping clusters at
    # x=100 (ft10/la16/abz5/ft20), x=225 (la36/la40) and x=300 (abz7/abz9/la31).
    label_off = {
        "abz7": (-15, 5), "abz9": (15, 3), "la31": (16, -11),
        "la36": (0, -13), "la40": (0, 8),
        "la16": (-16, -3), "abz5": (0, 8), "ft10": (16, -3), "ft20": (0, 8),
    }
    for s, y, n, w in zip(sizes, adv, names, win):
        c = PALETTE["geakg"] if w else PALETTE["ils"]
        ax.scatter([s], [y], s=80, color=c, edgecolor=PALETTE["ink"],
                   lw=0.7, zorder=4)
        dx, dy = label_off.get(n, (0, 7))
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        ax.annotate(n, (s, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.0, color=PALETTE["ink"], ha=ha, alpha=0.85)

    ax.set_xlabel("Problem size (operations $= n_{\\mathrm{jobs}}\\times n_{\\mathrm{machines}}$)")
    ax.set_ylabel("GEAKG advantage over ILS (%)")
    ax.set_xlim(0, sizes.max() + 25)

    # legend
    h = [plt.Line2D([], [], ls="", marker="o", mfc=PALETTE["geakg"],
                    mec=PALETTE["ink"], ms=8, label="GEAKG wins"),
         plt.Line2D([], [], ls="", marker="o", mfc=PALETTE["ils"],
                    mec=PALETTE["ink"], ms=8, label="ILS wins")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = "paper_els_cas/fig_transfer_crossover.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    print(f"GEAKG wins {sum(win)}/{len(win)}  | trend slope {b:+.3f}%/op  | crossover {xcross:.0f} ops")


if __name__ == "__main__":
    main()
