"""GEAKG paper figure palette — pink-primary, EMPATH-inspired.

Elegant soft pastels on a white background with dark ink, following the EMPATH
paper's aesthetic but with PINK as the primary/hero colour (GEAKG). Import and
call ``apply_paper_style()`` at the top of any figure script, then pull colours
from ``PALETTE``.
"""

import matplotlib.pyplot as plt

PALETTE = {
    "geakg":      "#D6336C",  # PRIMARY / hero — elegant rose-pink (GEAKG)
    "geakg_soft": "#FDC5F5",  # soft pink for fills / shaded regions (EMPATH)
    "ils":        "#8093F1",  # baseline (ILS) — periwinkle (EMPATH)
    "classic":    "#B3AACD",  # de-emphasised heuristics (SPT / LPT / GL)
    "accent":     "#B388EB",  # tertiary — lavender (EMPATH)
    "aqua":       "#72DDF7",  # quaternary — pearl aqua (EMPATH)
    "ink":        "#2F2A2B",  # text / axes / edges
    "grid":       "#EDE6EE",  # very light pink-lavender grid
}


def apply_paper_style():
    """Apply the pink-primary paper style to matplotlib rcParams."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
