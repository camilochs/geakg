#!/usr/bin/env python3
"""Generate an executive summary PDF for the GEAKG paper."""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
OUT_PDF = ROOT / "paper" / "GEAKG_Executive_Summary.pdf"

# ── colours ────────────────────────────────────────────────────────
C_BLUE   = "#3B82F6"   # L0 / CS1
C_GREEN  = "#22C55E"   # L1
C_ORANGE = "#F97316"   # L2 / CS2
C_GRAY   = "#6B7280"
C_DARK   = "#1F2937"
C_LIGHT  = "#F3F4F6"
C_ACCENT = "#8B5CF6"
C_RED    = "#EF4444"

# Case study specific
C_CS1     = "#2563EB"   # Deep blue for NAS
C_CS1_BG  = "#EFF6FF"
C_CS2     = "#EA580C"   # Deep orange for Optimization
C_CS2_BG  = "#FFF7ED"

PAGE_W, PAGE_H = 11, 8.5


def pdf_page_to_image(pdf_path: Path, page: int = 0, dpi: int = 600):
    """Convert a PDF page to a numpy array via PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pg = doc[page]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = pg.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    doc.close()
    return img


def _section_banner(fig, label, color, bg_color, y_top=0.97, h=0.05):
    """Draw a thin colored banner at the top of a page to identify the section."""
    ax = fig.add_axes([0, y_top - h, 1, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0", fc=bg_color, ec="none"))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 0.006, 1, boxstyle="square,pad=0", fc=color, ec="none"))
    ax.text(0.015, 0.5, label, fontsize=9, fontweight="bold",
            color=color, va="center", fontfamily="sans-serif")
    ax.axis("off")
    return ax


def _draw_table(ax, headers, rows, col_widths, header_color=C_DARK,
                highlight_col=None, row_stripe=True):
    """Draw a formatted table on an axes.

    col_widths: list of x positions for each column (cumulative left positions).
    rows: list of lists of strings.
    highlight_col: column index to highlight in green.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = len(rows)
    row_h = 0.85 / (n_rows + 1)
    y_start = 0.95

    # Header
    for c, h in enumerate(headers):
        ax.text(col_widths[c], y_start, h, fontsize=8, fontweight="bold",
                color=header_color, va="center", fontfamily="sans-serif")
    y_line = y_start - row_h * 0.5
    ax.plot([0, 1], [y_line, y_line], '-', color="#D1D5DB", lw=1)

    # Rows
    for r, row in enumerate(rows):
        y = y_start - (r + 1) * row_h
        if row_stripe and r % 2 == 1:
            ax.add_patch(mpatches.Rectangle(
                (0, y - row_h * 0.4), 1, row_h * 0.9,
                fc="#F9FAFB", ec="none", zorder=0))
        for c, cell in enumerate(row):
            weight = "normal"
            clr = C_DARK
            if highlight_col is not None and c == highlight_col:
                weight = "bold"
                clr = C_GREEN
            ax.text(col_widths[c], y, str(cell), fontsize=7.5, fontweight=weight,
                    color=clr, va="center", fontfamily="sans-serif")


# ===================================================================
# PAGE 1: TITLE
# ===================================================================
def add_title_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Header band
    ax_band = fig.add_axes([0, 0.82, 1, 0.18])
    ax_band.set_xlim(0, 1); ax_band.set_ylim(0, 1)
    ax_band.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0", fc=C_DARK, ec="none"))
    ax_band.text(0.5, 0.62, "GEAKG", fontsize=36, fontweight="bold",
                 color="white", ha="center", va="center", fontfamily="sans-serif")
    ax_band.text(0.5, 0.30,
                 "Generative Executable Algorithm Knowledge Graphs",
                 fontsize=15, color="#D1D5DB", ha="center", va="center",
                 fontfamily="sans-serif")
    ax_band.axis("off")

    # Authors
    ax = fig.add_axes([0.05, 0.74, 0.9, 0.06])
    ax.text(0.5, 0.5,
            "C. Chac\u00f3n Sartori, J. H. Garc\u00eda, A. V. Tomut, C. Blum",
            fontsize=10, color=C_GRAY, ha="center", va="center",
            fontfamily="sans-serif", style="italic")
    ax.axis("off")

    ax2 = fig.add_axes([0.05, 0.69, 0.9, 0.05])
    ax2.text(0.5, 0.5,
             "Executive Summary  \u2502  Key Results & Architecture Overview",
             fontsize=11, color=C_DARK, ha="center", va="center",
             fontweight="bold", fontfamily="sans-serif")
    ax2.axis("off")

    # What is GEAKG
    ax3 = fig.add_axes([0.06, 0.38, 0.88, 0.29])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_LIGHT, ec="#E5E7EB", lw=1))
    ax3.text(0.5, 0.92, "What is GEAKG?", fontsize=14, fontweight="bold",
             color=C_DARK, ha="center", va="top", fontfamily="sans-serif")
    desc = (
        "A new class of knowledge graphs where nodes store executable algorithmic\n"
        "operators, edges encode valid transitions with learned weights (ACO pheromones),\n"
        "and the graph is generative: it produces solutions when traversed.\n\n"
        "Three defining properties not jointly present in any prior KG paradigm:\n"
        "    Generative   \u2014  topology & operators synthesized by LLMs, not hand-crafted\n"
        "    Executable   \u2014  nodes contain runnable code, not descriptions\n"
        "    Transferable \u2014  meta-level knowledge persists & transfers zero-shot across domains"
    )
    ax3.text(0.5, 0.78, desc, fontsize=9.5, color=C_DARK, ha="center", va="top",
             fontfamily="monospace", linespacing=1.5)
    ax3.axis("off")

    # Two case study cards
    cases = [
        ("Case Study 1 \u2014 NAS",
         "Neural Architecture Search\n"
         "18 roles, 5 categories\n"
         "NAS-Bench-Graph (26K GNN, 9 datasets)\n"
         "NAS-Bench-201 (15.6K CNN, 3 datasets)\n"
         "Transfer: 70 cross-dataset pairs (0 tokens)\n"
         "Baselines: Random Search, RegEvo, BO",
         C_CS1, C_CS1_BG),
        ("Case Study 2 \u2014 Optimization",
         "Combinatorial Optimization\n"
         "11 roles, 3 categories\n"
         "TSP (TSPLIB, n=52\u20131002) \u2192 JSSP / QAP\n"
         "Baselines: LLaMEA (code-evolution),\n"
         "classical heuristics (SPT, GL)",
         C_CS2, C_CS2_BG),
    ]
    for i, (title, desc, color, bg) in enumerate(cases):
        x0 = 0.06 + i * 0.46
        ax_c = fig.add_axes([x0, 0.06, 0.42, 0.30])
        ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.03", fc=bg, ec=color, lw=2))
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 0.012, 1, boxstyle="square,pad=0", fc=color, ec="none"))
        ax_c.text(0.04, 0.93, title, fontsize=11, fontweight="bold",
                  color=color, va="top", fontfamily="sans-serif")
        ax_c.text(0.04, 0.48, desc, fontsize=8.5, color=C_DARK,
                  va="center", fontfamily="sans-serif", linespacing=1.5)
        ax_c.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 1b: TERMINOLOGY
# ===================================================================
def add_terminology_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Title
    ax_t = fig.add_axes([0.05, 0.91, 0.9, 0.07])
    ax_t.text(0.5, 0.5, "Key Terminology",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    ax_sub = fig.add_axes([0.05, 0.87, 0.9, 0.04])
    ax_sub.text(0.5, 0.5,
                "GEAKG bridges four fields. This glossary defines the core terms used throughout the paper.",
                fontsize=9.5, color=C_GRAY, ha="center", va="center",
                fontfamily="sans-serif", style="italic")
    ax_sub.axis("off")

    # Four category columns
    categories = [
        ("Knowledge Graphs", C_BLUE, "#DBEAFE", [
            ("Knowledge Graph (KG)",
             "A graph where nodes are entities and\nedges are typed relations between them."),
            ("Procedural KG",
             "A KG where nodes store executable\nprocedures, not declarative facts."),
            ("RoleSchema",
             "An ontology defining role categories\nand allowed transitions (the KG schema)."),
            ("Snapshot",
             "A serialized GEAKG (topology + operators\n+ learned weights) as a small JSON file."),
        ]),
        ("Optimization / ACO", C_ORANGE, "#FFF7ED", [
            ("Metaheuristic",
             "A high-level search strategy that guides\noperator selection (e.g., ACO, GA, ILS)."),
            ("ACO (Ant Colony Opt.)",
             "Swarm algorithm where artificial ants\ndeposit pheromones on good paths."),
            ("Pheromone",
             "A learned weight on a graph edge;\nhigher = more promising transition."),
            ("MMAS",
             "MAX\u2013MIN Ant System: ACO variant\nwith bounded pheromone values."),
        ]),
        ("Generative AI / LLMs", C_GREEN, "#DCFCE7", [
            ("LLM (Large Language Model)",
             "A neural model (e.g., GPT) that generates\ntext or code from natural-language prompts."),
            ("Code-Evolution",
             "Using LLMs to iteratively generate and\nrefine code (e.g., LLaMEA, FunSearch)."),
            ("L1 Operator",
             "An executable code snippet generated by\nan LLM and bound to a specific role."),
            ("Token Budget",
             "Total LLM tokens consumed during\noffline GEAKG construction."),
        ]),
        ("Software Engineering", C_ACCENT, "#F3E8FF", [
            ("Domain Binding (ctx)",
             "A protocol interface that a domain must\nimplement: evaluate(), valid(), decode()."),
            ("Symbolic Executor",
             "The runtime engine that traverses the\nGEAKG using rules\u2014no LLM calls."),
            ("Transfer (zero-shot)",
             "Applying a learned GEAKG to a new\ndomain by only swapping the binding."),
            ("Three-Layer Architecture",
             "L0 (topology), L1 (operators),\nL2 (learned pheromones + rules)."),
        ]),
    ]

    col_w = 0.22
    col_gap = 0.015
    x_start = 0.03
    y_top = 0.84

    for ci, (cat_name, color, bg, terms) in enumerate(categories):
        x = x_start + ci * (col_w + col_gap)

        # Category header
        ax_h = fig.add_axes([x, y_top, col_w, 0.04])
        ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1); ax_h.axis("off")
        ax_h.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.05",
            fc=color, ec="none"))
        ax_h.text(0.5, 0.5, cat_name, fontsize=9, fontweight="bold",
                  color="white", ha="center", va="center",
                  fontfamily="sans-serif")

        # Term cards
        card_h = 0.175
        card_gap = 0.012
        for ti, (term, desc) in enumerate(terms):
            y = y_top - 0.05 - ti * (card_h + card_gap)
            ax_c = fig.add_axes([x, y - card_h, col_w, card_h])
            ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off")
            ax_c.add_patch(mpatches.FancyBboxPatch(
                (0, 0), 1, 1, boxstyle="round,pad=0.03",
                fc=bg, ec=color, lw=1.0, alpha=0.8))
            ax_c.text(0.06, 0.82, term, fontsize=7.5, fontweight="bold",
                      color=color, va="top", fontfamily="sans-serif")
            ax_c.text(0.06, 0.52, desc, fontsize=6.5, color=C_DARK,
                      va="top", fontfamily="sans-serif", linespacing=1.3)

    # Bottom note
    ax_n = fig.add_axes([0.05, 0.02, 0.9, 0.04])
    ax_n.text(0.5, 0.5,
              "Bold terms appear frequently in the paper. "
              "The same GEAKG engine serves both case studies\u2014only the RoleSchema and domain binding change.",
              fontsize=8, color=C_GRAY, ha="center", va="center",
              fontfamily="sans-serif", style="italic")
    ax_n.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 2: PIPELINE
# ===================================================================
def add_pipeline_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    ax_t = fig.add_axes([0.05, 0.90, 0.9, 0.08])
    ax_t.text(0.5, 0.5, "End-to-End GEAKG Pipeline",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    # OFFLINE
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 2.8), 9.6, 4.0, boxstyle="round,pad=0.15",
        fc="#F0F9FF", ec=C_BLUE, lw=2, alpha=0.5))
    ax.text(5, 6.55, "OFFLINE PHASE (LLM + ACO)", fontsize=13,
            fontweight="bold", color=C_BLUE, ha="center", fontfamily="sans-serif")

    boxes = [
        ((0.5, 5.2, 2.0, 1.1), "RoleSchema", "Human-defined\nontology", "white", C_DARK),
        ((3.0, 5.2, 2.2, 1.1), "L0: MetaGraph", "Topology + transitions\n+ initial weights (LLM)", "#DBEAFE", C_BLUE),
        ((5.7, 5.2, 2.2, 1.1), "L1: Operator Pool", "Executable code per role\n+ validation (LLM)", "#DCFCE7", C_GREEN),
        ((3.5, 3.8, 3.0, 1.0), "L2: ACO Training", "Pheromones + symbolic rules", "#FFF7ED", C_ORANGE),
        ((7.0, 3.8, 2.5, 1.0), "GEAKG Snapshot", "L0+L1+L2  (~1-3 KB JSON)", "#F3F4F6", C_GRAY),
    ]
    for (bx, by, bw, bh), title, sub, fc, ec in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.5))
        ax.text(bx + bw/2, by + bh - 0.3, title, fontsize=10, fontweight="bold",
                color=ec if ec != C_DARK else C_DARK, ha="center", fontfamily="sans-serif")
        ax.text(bx + bw/2, by + 0.25, sub, fontsize=7.5, color=C_DARK,
                ha="center", fontfamily="sans-serif")

    # Arrows
    for (x1, x2, y) in [(2.6, 2.9, 5.75), (5.3, 5.6, 5.75), (6.6, 6.9, 4.3)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))
    ax.annotate("", xy=(5.0, 4.9), xytext=(5.0, 5.15),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))

    # Separator
    ax.plot([0.3, 9.7], [2.6, 2.6], '--', color=C_GRAY, lw=1.5, alpha=0.6)
    ax.text(5, 2.7, "deploy / transfer", fontsize=9, color=C_GRAY,
            ha="center", va="bottom", fontfamily="sans-serif", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    ax.annotate("", xy=(8.25, 2.5), xytext=(8.25, 3.7),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=C_ACCENT))

    # ONLINE
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 0.3), 9.6, 2.1, boxstyle="round,pad=0.15",
        fc="#FDF4FF", ec=C_ACCENT, lw=2, alpha=0.5))
    ax.text(5, 2.15, "ONLINE PHASE (Zero LLM Tokens)", fontsize=13,
            fontweight="bold", color=C_ACCENT, ha="center", fontfamily="sans-serif")

    for (bx, by, bw, bh), title, sub, ec in [
        ((1.0, 0.6, 3.5, 1.2), "Symbolic Executor",
         "Graph traversal + L2 rules\n+ pheromone-weighted selection", C_ACCENT),
        ((5.0, 0.6, 3.5, 1.2), "Domain Binding (ctx)",
         "evaluate(), valid(), decode()\nOnly part that changes per domain", C_GREEN),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.1", fc="white", ec=ec, lw=1.5))
        ax.text(bx + bw/2, by + bh - 0.35, title, fontsize=11, fontweight="bold",
                color=ec, ha="center", fontfamily="sans-serif")
        ax.text(bx + bw/2, by + 0.35, sub, fontsize=7.5, color=C_DARK,
                ha="center", fontfamily="sans-serif")

    ax.annotate("", xy=(4.8, 1.2), xytext=(4.6, 1.2),
                arrowprops=dict(arrowstyle="<->", lw=2, color=C_DARK))
    ax.text(9.2, 0.9, "0 LLM\ntokens", fontsize=11, fontweight="bold",
            color=C_RED, ha="center", va="center", fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=C_RED, lw=1.5))

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 2b: TOY EXAMPLE
# ===================================================================
def add_toy_example_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # ── Title ──
    ax_t = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "Putting It Together: A Toy Example",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # ── Subtitle ──
    ax_sub = fig.add_axes([0.05, 0.88, 0.9, 0.04])
    ax_sub.text(0.5, 0.5,
                "The complete GEAKG lifecycle on a 5-city routing problem with 3 roles.",
                fontsize=10, color=C_GRAY, ha="center", va="center",
                fontfamily="sans-serif", style="italic")
    ax_sub.axis("off")

    # ── Graph diagram (left, top half) ──
    ax_g = fig.add_axes([0.03, 0.48, 0.42, 0.38])
    ax_g.set_xlim(-0.5, 5.5)
    ax_g.set_ylim(-0.2, 3.5)
    ax_g.axis("off")

    # Constructive background
    ax_g.add_patch(mpatches.FancyBboxPatch(
        (-0.3, 0.5), 2.2, 2.5, boxstyle="round,pad=0.15",
        fc="#DCFCE7", ec=C_GREEN, lw=1.2, alpha=0.3))
    ax_g.text(0.8, 2.8, "Constructive", fontsize=8, fontweight="bold",
              color=C_GREEN, ha="center", fontfamily="sans-serif")

    # Improvement background
    ax_g.add_patch(mpatches.FancyBboxPatch(
        (2.8, 0.1), 2.5, 3.0, boxstyle="round,pad=0.15",
        fc="#DBEAFE", ec=C_BLUE, lw=1.2, alpha=0.3))
    ax_g.text(4.05, 2.9, "Improvement", fontsize=8, fontweight="bold",
              color=C_BLUE, ha="center", fontfamily="sans-serif")

    # Nodes
    for (cx, cy, label, fc, ec) in [
        (0.8, 1.6, "greedy_nn", "#DCFCE7", C_GREEN),
        (4.0, 2.3, "swap", "#DBEAFE", C_BLUE),
        (4.0, 0.9, "2opt", "#DBEAFE", C_BLUE),
    ]:
        ax_g.add_patch(mpatches.FancyBboxPatch(
            (cx - 0.7, cy - 0.25), 1.4, 0.5, boxstyle="round,pad=0.08",
            fc=fc, ec=ec, lw=1.8))
        ax_g.text(cx, cy, label, fontsize=9, fontweight="bold",
                  color=ec, ha="center", va="center", fontfamily="monospace")

    # Edge: greedy_nn -> swap (thin, dashed, low pheromone)
    ax_g.annotate("", xy=(3.3, 2.25), xytext=(1.5, 1.75),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_GRAY,
                                  linestyle="dashed"))
    ax_g.text(2.15, 2.25, "\u03c4 = 0.2", fontsize=8, color=C_GRAY,
              ha="center", fontfamily="sans-serif", style="italic",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Edge: greedy_nn -> 2opt (thick, high pheromone)
    ax_g.annotate("", xy=(3.3, 0.95), xytext=(1.5, 1.45),
                  arrowprops=dict(arrowstyle="-|>", lw=3.0, color=C_GREEN))
    ax_g.text(2.15, 0.95, "\u03c4 = 0.8", fontsize=8, color=C_GREEN,
              ha="center", fontweight="bold", fontfamily="sans-serif",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Edge: swap <-> 2opt (bidirectional)
    ax_g.annotate("", xy=(4.0, 1.2), xytext=(4.0, 2.0),
                  arrowprops=dict(arrowstyle="<->", lw=1.5, color=C_DARK))

    # ── Step-by-step cards (right, top half) ──
    # Each card: title on its own line above the description
    steps = [
        ("RoleSchema", C_DARK,
         "2 categories (Constructive, Improvement), 3 roles: greedy_nn, swap, 2opt\n"
         "Allowed: Constructive \u2192 Improvement; Improvement roles revisit each other"),
        ("L0 + L1 + L2", C_BLUE,
         "L0: 3 nodes, 4 edges. L1: 1 operator per role (few lines of code)\n"
         "L2: ACO on 5-city instance \u2192 greedy_nn\u21922opt gets 3\u00d7 more pheromone"),
        ("Snapshot", C_ORANGE,
         "Export as small JSON: 3 nodes, 4 weighted edges, 3 code snippets"),
        ("Transfer", C_ACCENT,
         "New domain (delivery scheduling): change only evaluate()\n"
         "Topology + learned pheromones transfer intact. 0 tokens, 0 retraining"),
    ]
    card_h = 0.085
    card_gap = 0.008
    card_top = 0.86
    for i, (title, color, desc) in enumerate(steps):
        y_bot = card_top - (i + 1) * card_h - i * card_gap
        ax_c = fig.add_axes([0.48, y_bot, 0.49, card_h])
        ax_c.set_xlim(0, 1)
        ax_c.set_ylim(0, 1)
        ax_c.axis("off")
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            fc="white", ec=color, lw=1.5))
        ax_c.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 0.012, 1, boxstyle="square,pad=0",
            fc=color, ec="none"))
        ax_c.text(0.03, 0.85, title, fontsize=9, fontweight="bold",
                  color=color, va="top", fontfamily="sans-serif")
        ax_c.text(0.03, 0.55, desc, fontsize=7, color=C_DARK,
                  va="top", fontfamily="sans-serif", linespacing=1.4)

    # ── Insight box (full width, middle) ──
    ax_b = fig.add_axes([0.05, 0.40, 0.90, 0.06])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    ax_b.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_LIGHT, ec="#E5E7EB", lw=1))
    ax_b.text(0.5, 0.5,
              "Learned insight: \"segment reversal (2opt) fixes more of the route "
              "per step than random swaps\" \u2014 domain-independent procedural knowledge.",
              fontsize=8.5, color=C_DARK, ha="center", va="center",
              fontfamily="sans-serif", style="italic")

    # ── Transfer illustration (full width, below insight) ──
    ax_tr = fig.add_axes([0.05, 0.22, 0.90, 0.16])
    ax_tr.set_xlim(0, 10)
    ax_tr.set_ylim(0, 2)
    ax_tr.axis("off")

    # Source domain
    ax_tr.add_patch(mpatches.FancyBboxPatch(
        (0.2, 0.3), 3.2, 1.4, boxstyle="round,pad=0.1",
        fc="#DCFCE7", ec=C_GREEN, lw=1.5))
    ax_tr.text(1.8, 1.45, "Routing (distance)", fontsize=9, fontweight="bold",
               color=C_GREEN, ha="center", va="center", fontfamily="sans-serif")
    ax_tr.text(1.8, 0.85, "evaluate() = total distance\n"
               "greedy_nn \u2192 2opt (\u03c4=0.8)",
               fontsize=7.5, color=C_DARK, ha="center", va="center",
               fontfamily="sans-serif", linespacing=1.4)

    # Arrow
    ax_tr.annotate("", xy=(5.5, 1.0), xytext=(3.6, 1.0),
                   arrowprops=dict(arrowstyle="-|>", lw=2.5, color=C_ACCENT))
    ax_tr.text(4.55, 1.45, "Transfer", fontsize=8, fontweight="bold",
               color=C_ACCENT, ha="center", va="center", fontfamily="sans-serif")
    ax_tr.text(4.55, 0.55, "change evaluate()\nonly", fontsize=7, color=C_ACCENT,
               ha="center", va="center", fontfamily="sans-serif", style="italic")

    # Target domain
    ax_tr.add_patch(mpatches.FancyBboxPatch(
        (5.7, 0.3), 3.2, 1.4, boxstyle="round,pad=0.1",
        fc="#FDF4FF", ec=C_ACCENT, lw=1.5))
    ax_tr.text(7.3, 1.45, "Delivery (time)", fontsize=9, fontweight="bold",
               color=C_ACCENT, ha="center", va="center", fontfamily="sans-serif")
    ax_tr.text(7.3, 0.85, "evaluate() = total time\n"
               "greedy_nn \u2192 2opt (\u03c4=0.8)",
               fontsize=7.5, color=C_DARK, ha="center", va="center",
               fontfamily="sans-serif", linespacing=1.4)

    # Zero tokens badge
    ax_tr.text(9.4, 1.0, "0 LLM\ntokens", fontsize=9, fontweight="bold",
               color=C_RED, ha="center", va="center", fontfamily="sans-serif",
               bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec=C_RED, lw=1.2))

    # ── Three properties (bottom) ──
    props = [
        ("Generative", "Topology could be\nLLM-produced", C_GREEN, "#DCFCE7"),
        ("Executable", "Every node runs\nreal code", C_BLUE, "#DBEAFE"),
        ("Transferable", "Snapshot applies to new\ndomain via binding swap", C_ORANGE, "#FFF7ED"),
    ]
    for i, (title, desc, color, bg) in enumerate(props):
        x_left = 0.05 + i * 0.31
        ax_pp = fig.add_axes([x_left, 0.04, 0.27, 0.15])
        ax_pp.set_xlim(0, 1)
        ax_pp.set_ylim(0, 1)
        ax_pp.axis("off")
        ax_pp.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.04",
            fc=bg, ec=color, lw=1.5))
        ax_pp.text(0.5, 0.72, title, fontsize=10, fontweight="bold",
                   color=color, ha="center", va="center", fontfamily="sans-serif")
        ax_pp.text(0.5, 0.30, desc, fontsize=7.5, color=C_DARK,
                   ha="center", va="center", fontfamily="sans-serif", linespacing=1.3)

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 3: METAGRAPHS (side by side)
# ===================================================================
def add_dual_metagraph_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    ax_t = fig.add_axes([0.05, 0.91, 0.9, 0.07])
    ax_t.text(0.5, 0.5, "GEAKG MetaGraph: Two Case Studies, Same Engine",
              fontsize=16, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # ── Left: NAS ──
    ax_nas = fig.add_axes([0.03, 0.12, 0.45, 0.75])
    ax_nas.set_xlim(0, 6); ax_nas.set_ylim(0, 8)
    ax_nas.axis("off")

    # NAS background panel
    ax_nas.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.3), 6.0, 7.5, boxstyle="round,pad=0.1",
        fc=C_CS1_BG, ec=C_CS1, lw=1.5, alpha=0.3))

    ax_nas.text(3, 7.5, "CASE STUDY 1", fontsize=8, fontweight="bold",
                color=C_CS1, ha="center", fontfamily="sans-serif", alpha=0.7)
    ax_nas.text(3, 7.1, "Neural Architecture Search", fontsize=12, fontweight="bold",
                color=C_CS1, ha="center", fontfamily="sans-serif")
    ax_nas.text(3, 6.7, "18 roles \u00b7 5 categories", fontsize=9, color=C_GRAY,
                ha="center", fontfamily="sans-serif")

    nas_cats = [
        ("Topology", 5.7, "#DBEAFE", C_CS1, ["cell_based", "feedforward", "recursive", "residual"]),
        ("Activation", 4.5, "#FEF3C7", "#D97706", ["standard", "mixed", "adaptive"]),
        ("Training", 3.3, "#DCFCE7", C_GREEN, ["optimizer", "loss", "scheduler", "augmentation"]),
        ("Regularization", 2.1, "#FFF7ED", C_ORANGE, ["dropout", "structural", "weight_decay"]),
        ("Evaluation", 0.9, "#F3E8FF", C_ACCENT, ["proxy", "full", "ensemble", "multi_metric"]),
    ]
    for cat_name, y, bg, ec, roles in nas_cats:
        ax_nas.add_patch(mpatches.FancyBboxPatch(
            (0.3, y), 5.4, 0.85, boxstyle="round,pad=0.08", fc=bg, ec=ec, lw=1.2, alpha=0.7))
        ax_nas.text(0.5, y + 0.65, cat_name, fontsize=8.5, fontweight="bold",
                    color=ec, va="center", fontfamily="sans-serif")
        ax_nas.text(3, y + 0.25, "  ".join(roles), fontsize=6.5, color=C_DARK,
                    ha="center", va="center", fontfamily="monospace")
    for i in range(len(nas_cats) - 1):
        y_from = nas_cats[i][1]
        y_to = nas_cats[i + 1][1] + 0.85
        ax_nas.annotate("", xy=(3, y_to), xytext=(3, y_from),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=C_GRAY))

    # ── Right: Optimization ──
    ax_opt = fig.add_axes([0.52, 0.12, 0.45, 0.75])
    ax_opt.set_xlim(0, 6); ax_opt.set_ylim(0, 8)
    ax_opt.axis("off")

    ax_opt.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.3), 6.0, 7.5, boxstyle="round,pad=0.1",
        fc=C_CS2_BG, ec=C_CS2, lw=1.5, alpha=0.3))

    ax_opt.text(3, 7.5, "CASE STUDY 2", fontsize=8, fontweight="bold",
                color=C_CS2, ha="center", fontfamily="sans-serif", alpha=0.7)
    ax_opt.text(3, 7.1, "Combinatorial Optimization", fontsize=12, fontweight="bold",
                color=C_CS2, ha="center", fontfamily="sans-serif")
    ax_opt.text(3, 6.7, "11 roles \u00b7 3 categories", fontsize=9, color=C_GRAY,
                ha="center", fontfamily="sans-serif")

    opt_cats = [
        ("Construction", 5.0, "#DBEAFE", C_BLUE,
         ["greedy", "insertion", "savings", "random"]),
        ("Local Search", 3.2, "#DCFCE7", C_GREEN,
         ["intensify_small", "intensify_med", "intensify_large", "diversify"]),
        ("Perturbation", 1.4, "#FFF7ED", C_ORANGE,
         ["controlled", "aggressive", "random"]),
    ]
    for cat_name, y, bg, ec, roles in opt_cats:
        ax_opt.add_patch(mpatches.FancyBboxPatch(
            (0.3, y), 5.4, 1.3, boxstyle="round,pad=0.08", fc=bg, ec=ec, lw=1.2, alpha=0.7))
        ax_opt.text(0.5, y + 1.05, cat_name, fontsize=8.5, fontweight="bold",
                    color=ec, va="center", fontfamily="sans-serif")
        ax_opt.text(3, y + 0.4, "  ".join(roles), fontsize=6.5, color=C_DARK,
                    ha="center", va="center", fontfamily="monospace")

    ax_opt.annotate("", xy=(3, 4.95), xytext=(3, 5.0 - 0.15),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=C_GRAY))
    ax_opt.annotate("", xy=(3, 2.65), xytext=(3, 3.15),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=C_GRAY))
    ax_opt.annotate("", xy=(5.5, 5.0), xytext=(5.5, 2.1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=C_GRAY,
                                   linestyle="dashed", connectionstyle="arc3,rad=0.3"))
    ax_opt.text(5.9, 3.5, "re-optimize", fontsize=7, color=C_GRAY,
                rotation=90, ha="center", va="center", fontfamily="sans-serif",
                style="italic")

    # Bottom note
    ax_note = fig.add_axes([0.05, 0.03, 0.9, 0.07])
    ax_note.text(0.5, 0.5,
                 "Both MetaGraphs are traversed by the identical ACO engine. "
                 "No framework code differs between case studies \u2014 only the RoleSchema changes.",
                 fontsize=9.5, color=C_DARK, ha="center", va="center",
                 fontfamily="sans-serif", style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT, ec="#E5E7EB"))
    ax_note.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 4: RQ + SUMMARY
# ===================================================================
def add_rq_results_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    ax_t = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "Research Questions & Key Results",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    rqs = [
        ("RQ1: Generality",
         "Can GEAKG serve fundamentally different\ndomains without framework code changes?",
         "NAS (18 roles) + Optimization (11 roles)\nsame engine, different RoleSchema only",
         C_BLUE),
        ("RQ2: Transferability",
         "Can knowledge learned on one domain\ntransfer to unseen domains?",
         "TSP \u2192 JSSP, QAP (zero-shot)\n70/70 NAS transfers win vs Random",
         C_GREEN),
        ("RQ3: Persistence",
         "Does GEAKG enable zero-shot transfer\nat near-zero marginal cost?",
         "~1-3 KB snapshot, 0 tokens online\n50k vs 150k tokens (3 domains)",
         C_ORANGE),
    ]
    for i, (title, question, answer, color) in enumerate(rqs):
        y = 0.74 - i * 0.175
        ax_rq = fig.add_axes([0.05, y, 0.90, 0.155])
        ax_rq.set_xlim(0, 1); ax_rq.set_ylim(0, 1)
        ax_rq.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02", fc="white", ec=color, lw=2))
        ax_rq.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 0.008, 1, boxstyle="square,pad=0", fc=color, ec="none"))
        ax_rq.text(0.03, 0.82, title, fontsize=11, fontweight="bold",
                   color=color, va="top", fontfamily="sans-serif")
        ax_rq.text(0.03, 0.42, question, fontsize=8.5, color=C_DARK,
                   va="center", fontfamily="sans-serif", style="italic")
        ax_rq.text(0.55, 0.42, answer, fontsize=8.5, color=C_DARK,
                   va="center", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT, ec="#E5E7EB"))
        ax_rq.axis("off")

    # Summary table
    ax_tt = fig.add_axes([0.05, 0.195, 0.9, 0.04])
    ax_tt.text(0.5, 0.5, "Summary of Key Quantitative Results",
               fontsize=13, fontweight="bold", color=C_DARK,
               ha="center", va="center", fontfamily="sans-serif")
    ax_tt.axis("off")

    ax_tab = fig.add_axes([0.06, 0.03, 0.88, 0.17])
    ax_tab.axis("off"); ax_tab.set_xlim(0, 1); ax_tab.set_ylim(0, 1)

    data = [
        ["Metric", "CS1: NAS", "CS2: Optimization"],
        ["Primary baselines", "Random, RegEvo, BO (GP+EI)", "LLaMEA-50k (code-evolution)"],
        ["Win rate", "100% (70/70 vs Random)", "5/7 TSP instances (25\u201365% gains)"],
        ["Statistical significance", "89% at \u03b1=0.05", "p<0.05 on n \u2265 150"],
        ["Transfer cost", "0 tokens (snapshot)", "0 tokens (binding swap)"],
        ["Robustness", "1.3\u00d7\u20134.8\u00d7 lower variance", "100% success (all LLM tiers)"],
    ]
    cs_colors = [C_DARK, C_CS1, C_CS2]
    for r, row in enumerate(data):
        y_pos = 0.92 - r * 0.16
        for c, cell in enumerate(row):
            x = 0.01 + c * 0.34
            weight = "bold" if r == 0 else "normal"
            clr = cs_colors[c] if r > 0 else C_DARK
            ax_tab.text(x, y_pos, cell, fontsize=8.5,
                        fontweight=weight, color=clr, va="center",
                        fontfamily="sans-serif")
        if r == 0:
            ax_tab.plot([0.01, 0.99], [y_pos - 0.07, y_pos - 0.07],
                        '-', color="#D1D5DB", lw=1)

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 5: CS2 — TSP RESULTS TABLE
# ===================================================================
def add_cs2_tsp_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _section_banner(fig, "CASE STUDY 2 \u2014 Combinatorial Optimization", C_CS2, C_CS2_BG)

    ax_t = fig.add_axes([0.05, 0.85, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "TSP: GEAKG(LLaMEA-15k) vs LLaMEA-50k  (gpt5.2)",
              fontsize=15, fontweight="bold", color=C_CS2,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # Table
    ax_tab = fig.add_axes([0.06, 0.38, 0.88, 0.44])
    headers = ["Instance", "n", "GEAKG (15k tokens)", "LLaMEA (50k tokens)", "Improv."]
    rows = [
        ["berlin52",  "52",   "0.031 \u00b1 0.00",  "0.031 \u00b1 0.00",  "\u2014"],
        ["kroA100",  "100",  "0.025 \u00b1 0.02",  "0.016 \u00b1 0.00",  "\u2014"],
        ["ch150",    "150",  "0.578 \u00b1 0.18",  "0.828 \u00b1 0.26",  "30%"],
        ["pr226",    "226",  "0.628 \u00b1 0.27",  "1.810 \u00b1 0.00",  "65%"],
        ["pcb442",   "442",  "3.444 \u00b1 0.37",  "7.408 \u00b1 0.00",  "54%"],
        ["rat783",   "783",  "9.158 \u00b1 2.89", "12.246 \u00b1 0.16", "25%"],
        ["pr1002",  "1002", "8.880 \u00b1 2.41", "12.197 \u00b1 0.65", "27%"],
    ]
    _draw_table(ax_tab, headers, rows,
                col_widths=[0.0, 0.12, 0.22, 0.52, 0.85],
                header_color=C_CS2, highlight_col=2)

    # Summary box
    ax_s = fig.add_axes([0.06, 0.26, 0.88, 0.10])
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
    ax_s.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_CS2_BG, ec=C_CS2, lw=1.5))
    ax_s.text(0.5, 0.7,
              "GEAKG wins 5/7 instances  \u00b7  25\u201365% improvement on n \u2265 150  "
              "\u00b7  Uses 3.3\u00d7 fewer tokens",
              fontsize=10, fontweight="bold", color=C_CS2, ha="center", va="center",
              fontfamily="sans-serif")
    ax_s.text(0.5, 0.25,
              "Budget is intentionally asymmetric (15k vs 50k): GEAKG's structural knowledge reduces the token budget needed.",
              fontsize=8, color=C_GRAY, ha="center", va="center",
              fontfamily="sans-serif", style="italic")

    # Success rate table
    ax_t2 = fig.add_axes([0.06, 0.17, 0.88, 0.06])
    ax_t2.text(0.5, 0.5, "Success Rate by LLM Capability",
               fontsize=12, fontweight="bold", color=C_DARK,
               ha="center", va="center", fontfamily="sans-serif")
    ax_t2.axis("off")

    ax_sr = fig.add_axes([0.06, 0.03, 0.88, 0.14])
    sr_headers = ["LLM", "Budget", "GEAKG", "LLaMEA"]
    sr_rows = [
        ["gpt5.2",         "50k", "7/7 (100%)", "7/7 (100%)"],
        ["gpt-4o-mini",    "10k", "7/7 (100%)", "4/7 (57%)"],
        ["gpt-4o-mini",    "50k", "7/7 (100%)", "4/7 (57%)"],
        ["Qwen2.5-14B",   "10k", "7/7 (100%)", "1/7 (14%)"],
    ]
    _draw_table(ax_sr, sr_headers, sr_rows,
                col_widths=[0.0, 0.25, 0.45, 0.72],
                header_color=C_DARK, highlight_col=2)

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 5b: WHY THESE BASELINES + ABSORPTION ARGUMENT
# ===================================================================
def add_baseline_rationale_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    ax_t = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "Why These Baselines? Experimental Rationale",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # ── CS1 rationale ──
    ax_cs1 = fig.add_axes([0.05, 0.60, 0.90, 0.30])
    ax_cs1.set_xlim(0, 1); ax_cs1.set_ylim(0, 1)
    ax_cs1.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_CS1_BG, ec=C_CS1, lw=2))
    ax_cs1.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 0.008, 1, boxstyle="square,pad=0", fc=C_CS1, ec="none"))
    ax_cs1.text(0.025, 0.95, "CS1: Neural Architecture Search", fontsize=11,
                fontweight="bold", color=C_CS1, va="top", fontfamily="sans-serif")

    # Benchmarks
    ax_cs1.text(0.025, 0.80, "Benchmarks", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs1.text(0.12, 0.80,
                "NAS-Bench-Graph (26K GNN, 9 datasets) + NAS-Bench-201 (15.6K CNN, 3 datasets).\n"
                "Tabular benchmarks with pre-computed quality \u2192 reproducible, no GPU noise.",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif", linespacing=1.4)

    # Transfer
    ax_cs1.text(0.025, 0.62, "Transfer", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs1.text(0.12, 0.62,
                "Cross-dataset: 64 pairs (8 src \u00d7 8 tgt) on NAS-Bench-Graph + "
                "6 pairs (3 src \u00d7 2 tgt) on NAS-Bench-201 = 70 total.\n"
                "Pheromone snapshot learned on one dataset transfers to another "
                "(e.g., Cora \u2192 Photo) at 0 tokens.",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif", linespacing=1.4)

    # Baselines as compact table
    ax_cs1.text(0.025, 0.42, "Baselines", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    baselines_cs1 = [
        ("Random Search", "Same operator pool, random order \u2192 pure sequence ablation"),
        ("RegEvo", "De facto evolutionary NAS baseline (pop=50, tournament=10)"),
        ("Bayesian Opt.", "GP + EI surrogate; time-matched (~2s) and unlimited (500 evals, ~22 min)"),
    ]
    for i, (name, desc) in enumerate(baselines_cs1):
        y = 0.33 - i * 0.10
        ax_cs1.text(0.025, y, "\u2022 " + name, fontsize=7.5, fontweight="bold",
                    color=C_CS1, va="top", fontfamily="sans-serif")
        ax_cs1.text(0.18, y, desc, fontsize=7.5, color=C_DARK,
                    va="top", fontfamily="sans-serif")

    # Key result
    ax_cs1.text(0.025, 0.05, "Key:",  fontsize=7.5, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs1.text(0.065, 0.05,
                "100% win rate (70/70) vs Random. "
                "Time-matched BO: ~25% acc vs GEAKG's ~76%. "
                "Unlimited BO: <1 pp gain at 300\u20134600\u00d7 cost.",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs1.axis("off")

    # ── CS2 rationale ──
    ax_cs2 = fig.add_axes([0.05, 0.22, 0.90, 0.35])
    ax_cs2.set_xlim(0, 1); ax_cs2.set_ylim(0, 1)
    ax_cs2.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_CS2_BG, ec=C_CS2, lw=2))
    ax_cs2.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 0.008, 1, boxstyle="square,pad=0", fc=C_CS2, ec="none"))
    ax_cs2.text(0.025, 0.96, "CS2: Combinatorial Optimization", fontsize=11,
                fontweight="bold", color=C_CS2, va="top", fontfamily="sans-serif")

    # Benchmarks
    ax_cs2.text(0.025, 0.85, "Benchmarks", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs2.text(0.12, 0.85,
                "TSP source: TSPLIB instances (berlin52, kroA100, ch150, pr226, pcb442, rat783, pr1002; n=52\u20131002).\n"
                "Transfer targets: JSSP (Fisher-Thompson, Lawrence, Taillard; 6\u00d76 to 50\u00d715),\n"
                "QAP (QAPLIB; n=12\u2013256).",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif", linespacing=1.4)

    # Transfer
    ax_cs2.text(0.025, 0.64, "Transfer", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs2.text(0.12, 0.64,
                "Cross-domain: full GEAKG snapshot (L0+L1+L2) learned on TSP transfers zero-shot\n"
                "to 2 target domains. Only the domain binding changes \u2014 0 LLM tokens per transfer.",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif", linespacing=1.4)

    # Baselines
    ax_cs2.text(0.025, 0.47, "Baselines", fontsize=8, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    baselines_cs2 = [
        ("LLaMEA (TSP)", "state-of-art LLM code-evolution; strongest automated operator design"),
        ("SPT / LPT (JSSP)", "shortest/longest processing time dispatching rules"),
        ("Gilmore-Lawler (QAP)", "lower-bound + greedy assignment (1962)"),
    ]
    for i, (name, desc) in enumerate(baselines_cs2):
        y = 0.39 - i * 0.07
        ax_cs2.text(0.025, y, "\u2022 " + name, fontsize=7.5, fontweight="bold",
                    color=C_CS2, va="top", fontfamily="sans-serif")
        ax_cs2.text(0.24, y, desc, fontsize=7.5, color=C_DARK,
                    va="top", fontfamily="sans-serif")

    # Key
    ax_cs2.text(0.025, 0.04, "Key:",  fontsize=7.5, fontweight="bold",
                color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs2.text(0.065, 0.04,
                "GEAKG(LLaMEA-15k) wins 5/7 TSP vs LLaMEA-50k. "
                "JSSP: wins 8/14. QAP: dominates at n\u2265150 (ILS degrades from 4% to 17%).",
                fontsize=7.5, color=C_DARK, va="top", fontfamily="sans-serif")
    ax_cs2.axis("off")

    # ── Absorption argument box ──
    ax_abs = fig.add_axes([0.05, 0.04, 0.90, 0.15])
    ax_abs.set_xlim(0, 1); ax_abs.set_ylim(0, 1)
    ax_abs.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc="#FDF4FF", ec=C_ACCENT, lw=2))

    ax_abs.text(0.5, 0.88, "Why GEAKG(LLaMEA) > standalone LLaMEA",
                fontsize=10, fontweight="bold", color=C_ACCENT, ha="center",
                va="top", fontfamily="sans-serif")

    abs_items = [
        ("\u2022 LLaMEA alone:",
         "generates operators but must discover composition order implicitly (50k tokens)"),
        ("\u2022 GEAKG absorbs:",
         "operator into typed graph; topology constrains which, ACO learns when (15k tokens)"),
        ("\u2022 Result:",
         "3.3\u00d7 fewer tokens, 25\u201365% better on large instances, 100% success on all LLM tiers"),
    ]
    for i, (label, text) in enumerate(abs_items):
        y = 0.65 - i * 0.22
        ax_abs.text(0.03, y, label, fontsize=7.5, fontweight="bold",
                    color=C_ACCENT, va="top", fontfamily="sans-serif")
        ax_abs.text(0.20, y, text, fontsize=7.5, color=C_DARK,
                    va="top", fontfamily="sans-serif")
    ax_abs.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 6: CS2 — TRANSFER RESULTS
# ===================================================================
def add_cs2_transfer_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _section_banner(fig, "CASE STUDY 2 \u2014 Combinatorial Optimization", C_CS2, C_CS2_BG)

    ax_t = fig.add_axes([0.05, 0.85, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "Zero-Shot Cross-Domain Transfer (0 LLM tokens)",
              fontsize=15, fontweight="bold", color=C_CS2,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # JSSP table (top half)
    ax_jt = fig.add_axes([0.06, 0.72, 0.88, 0.10])
    ax_jt.text(0.0, 0.5, "TSP \u2192 JSSP (Job Shop Scheduling)",
               fontsize=11, fontweight="bold", color=C_CS2,
               va="center", fontfamily="sans-serif")
    ax_jt.axis("off")

    ax_jssp = fig.add_axes([0.06, 0.44, 0.88, 0.27])
    jssp_h = ["Instance", "Size", "GEAKG", "ILS", "SPT", "LPT"]
    jssp_r = [
        ["ft06",  "6\u00d76",    "0.00",         "0.00",          "98.18",  "134.55"],
        ["la16",  "10\u00d710",  "3.66 \u00b1 0.17", "4.80 \u00b1 2.11",  "265.71", "229.63"],
        ["abz5",  "10\u00d710",  "2.25 \u00b1 0.65", "4.69 \u00b1 3.95",  "278.28", "277.47"],
        ["orb01", "10\u00d710",  "8.90 \u00b1 2.20", "11.46 \u00b1 2.36", "138.15", "201.98"],
        ["ta21",  "20\u00d720", "29.94 \u00b1 6.10", "397.26",           "613.40", "609.68"],
        ["ta41",  "30\u00d720", "38.78 \u00b1 6.59", "467.03",           "795.76", "851.00"],
        ["ta51",  "50\u00d715", "25.19 \u00b1 7.41", "421.34",           "649.67", "694.46"],
    ]
    _draw_table(ax_jssp, jssp_h, jssp_r,
                col_widths=[0.0, 0.11, 0.24, 0.46, 0.68, 0.84],
                header_color=C_CS2, highlight_col=2)

    # QAP table (bottom half)
    ax_qt = fig.add_axes([0.06, 0.35, 0.88, 0.07])
    ax_qt.text(0.0, 0.5, "TSP \u2192 QAP (Quadratic Assignment Problem)",
               fontsize=11, fontweight="bold", color=C_CS2,
               va="center", fontfamily="sans-serif")
    ax_qt.axis("off")

    ax_qap = fig.add_axes([0.06, 0.12, 0.88, 0.22])
    qap_h = ["Instance", "n", "GEAKG", "ILS", "GL", "Winner"]
    qap_r = [
        ["nug12",   "12",  "0.00", "0.00",  "25.26", "Tie"],
        ["nug25",   "25",  "0.25", "0.06",  "36.06", "ILS"],
        ["tai50a",  "50",  "3.91", "3.29",  "18.93", "ILS"],
        ["tai100a", "100", "6.47", "4.09",  "14.20", "ILS"],
        ["tai150b", "150", "7.05", "13.79", "30.36", "GEAKG"],
        ["tai256c", "256", "3.73", "17.18", "120.48", "GEAKG"],
    ]
    _draw_table(ax_qap, qap_h, qap_r,
                col_widths=[0.0, 0.10, 0.22, 0.40, 0.58, 0.76],
                header_color=C_CS2, highlight_col=2)

    # Key insight
    ax_ins = fig.add_axes([0.06, 0.03, 0.88, 0.07])
    ax_ins.set_xlim(0, 1); ax_ins.set_ylim(0, 1); ax_ins.axis("off")
    ax_ins.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_CS2_BG, ec=C_CS2, lw=1.5))
    ax_ins.text(0.5, 0.5,
                "JSSP: GEAKG wins 8/14 \u00b7 QAP: GEAKG dominates at n \u2265 150 (ILS degrades from 4% to 17%)  "
                "\u00b7  All transfers use 0 tokens",
                fontsize=9, fontweight="bold", color=C_CS2, ha="center", va="center",
                fontfamily="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 7: CS2 — COST COMPARISON
# ===================================================================
def add_cost_comparison_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _section_banner(fig, "CASE STUDY 2 \u2014 Combinatorial Optimization", C_CS2, C_CS2_BG)

    ax_t = fig.add_axes([0.05, 0.85, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "Token Cost: GEAKG vs Code-Evolution (3 Domains)",
              fontsize=15, fontweight="bold", color=C_CS2,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    ax = fig.add_axes([0.12, 0.18, 0.76, 0.62])
    domains = ["TSP\n(source)", "JSSP", "QAP", "Total"]
    llamea = [50000, 50000, 50000, 150000]
    geakg  = [50000, 0, 0, 50000]
    x = np.arange(len(domains))
    w = 0.35

    ax.bar(x - w/2, llamea, w, label="LLaMEA (per domain)",
           color=C_RED, alpha=0.8, edgecolor="white", lw=1)
    ax.bar(x + w/2, geakg, w, label="GEAKG (transfer)",
           color=C_GREEN, alpha=0.8, edgecolor="white", lw=1)
    ax.set_ylabel("LLM Tokens", fontsize=12, fontfamily="sans-serif")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10, fontfamily="sans-serif")
    ax.legend(fontsize=11, loc="upper left", frameon=True)
    ax.set_ylim(0, 180000)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))

    for i in range(1, 3):
        ax.text(x[i] + w/2, 3000, "0", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_GREEN, fontfamily="sans-serif")
    ax.text(x[3] - w/2, llamea[3] + 5000, "150k", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=C_RED, fontfamily="sans-serif")
    ax.text(x[3] + w/2, geakg[3] + 5000, "50k", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=C_GREEN, fontfamily="sans-serif")
    ax.text(2.0, 130000, "3\u00d7 fewer tokens\nwith GEAKG transfer",
            fontsize=14, fontweight="bold", color=C_GREEN, ha="center",
            fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.4", fc="#DCFCE7", ec=C_GREEN, lw=1.5))

    ax_c = fig.add_axes([0.06, 0.04, 0.88, 0.08])
    ax_c.text(0.5, 0.5,
              "LLaMEA requires ~50k tokens per domain (150k total for 3 domains). "
              "GEAKG learns on TSP once (50k tokens), then transfers zero-shot to "
              "JSSP, QAP at zero token cost.",
              fontsize=9, color=C_GRAY, ha="center", va="center",
              fontfamily="sans-serif", wrap=True)
    ax_c.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 8: CS1 — NAS AGGREGATE RESULTS
# ===================================================================
def add_cs1_results_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    _section_banner(fig, "CASE STUDY 1 \u2014 Neural Architecture Search", C_CS1, C_CS1_BG)

    ax_t = fig.add_axes([0.05, 0.85, 0.9, 0.06])
    ax_t.text(0.5, 0.5, "NAS: Aggregate Transfer Statistics",
              fontsize=15, fontweight="bold", color=C_CS1,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    # NAS-Bench-Graph table
    ax_nt = fig.add_axes([0.06, 0.72, 0.88, 0.10])
    ax_nt.text(0.0, 0.5, "NAS-Bench-Graph  (8 sources \u00d7 8 targets = 64 pairs, 26K GNN architectures)",
               fontsize=10, fontweight="bold", color=C_CS1,
               va="center", fontfamily="sans-serif")
    ax_nt.axis("off")

    ax_ng = fig.add_axes([0.06, 0.56, 0.88, 0.15])
    ng_h = ["Metric", "vs Random Search", "vs RegEvo"]
    ng_r = [
        ["Wins (mean)",             "64/64 (100%)", "39/64 (61%)"],
        ["Significant (p < 0.05)",  "57/64 (89%)",  "10/64 (16%)"],
        ["Wall-time per transfer",  "~0.1s (0 tokens)", "~0.1s (0 tokens)"],
    ]
    _draw_table(ax_ng, ng_h, ng_r,
                col_widths=[0.0, 0.35, 0.68],
                header_color=C_CS1, highlight_col=1)

    # NAS-Bench-201 table
    ax_nt2 = fig.add_axes([0.06, 0.46, 0.88, 0.08])
    ax_nt2.text(0.0, 0.5, "NAS-Bench-201  (3 sources \u00d7 2 targets = 6 pairs, 15.6K CNN cells)",
                fontsize=10, fontweight="bold", color=C_CS1,
                va="center", fontfamily="sans-serif")
    ax_nt2.axis("off")

    ax_n2 = fig.add_axes([0.06, 0.28, 0.88, 0.17])
    n2_h = ["Metric", "vs Random Search", "vs RegEvo"]
    n2_r = [
        ["Wins (mean)",             "6/6 (100%)",  "4/6 (67%)"],
        ["Significant (p < 0.05)",  "5/6 (83%)",   "0/6 (0%)"],
        ["Mean \u0394 accuracy",    "+0.84 pp",    "+0.06 pp"],
        ["Wall-time per transfer",  "~1.8s (0 tokens)", "~1.8s (0 tokens)"],
    ]
    _draw_table(ax_n2, n2_h, n2_r,
                col_widths=[0.0, 0.35, 0.68],
                header_color=C_CS1, highlight_col=1)

    # Summary
    ax_s = fig.add_axes([0.06, 0.17, 0.88, 0.08])
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
    ax_s.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02", fc=C_CS1_BG, ec=C_CS1, lw=1.5))
    ax_s.text(0.5, 0.5,
              "70/70 pairs won (100% win rate)  \u00b7  89% statistically significant  "
              "\u00b7  1.3\u00d7\u20134.8\u00d7 lower variance  \u00b7  0 LLM tokens at deployment",
              fontsize=9.5, fontweight="bold", color=C_CS1, ha="center", va="center",
              fontfamily="sans-serif")

    # Cost analysis
    ax_ct = fig.add_axes([0.06, 0.09, 0.88, 0.06])
    ax_ct.text(0.0, 0.5, "NAS Cost Analysis",
               fontsize=10, fontweight="bold", color=C_DARK,
               va="center", fontfamily="sans-serif")
    ax_ct.axis("off")

    ax_cost = fig.add_axes([0.06, 0.03, 0.88, 0.06])
    ax_cost.set_xlim(0, 1); ax_cost.set_ylim(0, 1); ax_cost.axis("off")
    costs = [
        ("L1 Pool generation:", "~15k tokens (one-time)"),
        ("Snapshot size:", "1\u20133 KB JSON"),
        ("Transfer deployment:", "0 tokens, <2s"),
    ]
    for i, (label, value) in enumerate(costs):
        x = 0.02 + i * 0.34
        ax_cost.text(x, 0.5, label, fontsize=8, color=C_GRAY, va="center",
                     fontfamily="sans-serif")
        ax_cost.text(x + 0.16, 0.5, value, fontsize=8, fontweight="bold",
                     color=C_CS1, va="center", fontfamily="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 9: CS1 — NAS HEATMAP (from paper figure)
# ===================================================================
def add_figure_page(pdf, fig_path, title, caption, banner_label=None,
                    banner_color=None, banner_bg=None):
    if not fig_path.exists():
        print(f"  [SKIP] {fig_path.name} not found")
        return

    img = pdf_page_to_image(fig_path, dpi=600)
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    if banner_label:
        _section_banner(fig, banner_label, banner_color, banner_bg)

    ax_t = fig.add_axes([0.05, 0.85, 0.9, 0.06])
    ax_t.text(0.5, 0.5, title, fontsize=15, fontweight="bold",
              color=banner_color or C_DARK, ha="center", va="center",
              fontfamily="sans-serif")
    ax_t.axis("off")

    ax_img = fig.add_axes([0.06, 0.12, 0.88, 0.72])
    ax_img.imshow(img, interpolation="lanczos")
    ax_img.axis("off")

    ax_c = fig.add_axes([0.06, 0.02, 0.88, 0.09])
    ax_c.text(0.5, 0.5, caption, fontsize=8.5, color=C_GRAY,
              ha="center", va="center", fontfamily="sans-serif",
              wrap=True, linespacing=1.4)
    ax_c.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# PAGE 11: CONCLUSIONS
# ===================================================================
def add_conclusions_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    ax_t = fig.add_axes([0.05, 0.91, 0.9, 0.07])
    ax_t.text(0.5, 0.5, "Conclusions & Impact",
              fontsize=18, fontweight="bold", color=C_DARK,
              ha="center", va="center", fontfamily="sans-serif")
    ax_t.axis("off")

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.80])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    takeaways = [
        ("New KG Paradigm",
         "GEAKG introduces procedural knowledge graphs where nodes are executable,\n"
         "edges encode learned composition patterns, and the graph generates solutions.",
         C_BLUE),
        ("Domain-Agnostic Engine",
         "The same 3-layer architecture (L0/L1/L2) + ACO engine serves NAS and\n"
         "combinatorial optimization by swapping only the RoleSchema ontology.",
         C_GREEN),
        ("Zero-Cost Transfer",
         "Knowledge persists in ~1-3 KB JSON snapshots. Transfer to new domains\n"
         "requires only changing the domain binding \u2014 0 LLM tokens at runtime.",
         C_ORANGE),
        ("Empirical Validation",
         "CS1 (NAS): 100% win rate on 70 pairs, 89% significant.\n"
         "CS2 (Opt): 5/7 on TSP vs LLaMEA-50k, transfer to 2 domains at 0 cost.",
         C_ACCENT),
        ("Knowledge Engineering Bridge",
         "GEAKG supports a complete knowledge lifecycle: acquisition (LLM),\n"
         "validation, refinement (ACO), persistence (snapshot), and transfer.",
         C_RED),
    ]
    for i, (title, desc, color) in enumerate(takeaways):
        y = 0.88 - i * 0.175
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.07), 1.0, 0.15, boxstyle="round,pad=0.02",
            fc="white", ec=color, lw=1.5))
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.07), 0.006, 0.15, boxstyle="square,pad=0",
            fc=color, ec="none"))
        ax.text(0.025, y + 0.05, title, fontsize=11, fontweight="bold",
                color=color, va="center", fontfamily="sans-serif")
        ax.text(0.025, y - 0.025, desc, fontsize=8.5, color=C_DARK,
                va="center", fontfamily="sans-serif", linespacing=1.4)

    ax.text(0.5, 0.03,
            '\u201cProcedural knowledge can be explicitly represented, learned, and transferred\n'
            'via executable knowledge graphs.\u201d',
            fontsize=10, color=C_GRAY, ha="center", va="center",
            fontfamily="sans-serif", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("Generating GEAKG Executive Summary...")
    n = 0

    with PdfPages(str(OUT_PDF)) as pdf:
        pages = [
            ("Title page",          add_title_page),
            ("Terminology",         add_terminology_page),
            ("Pipeline",            add_pipeline_page),
            ("Toy example",         add_toy_example_page),
            ("MetaGraphs",          add_dual_metagraph_page),
            ("RQs & summary",       add_rq_results_page),
            ("Baseline rationale",  add_baseline_rationale_page),
            ("CS2: TSP results",    add_cs2_tsp_page),
            ("CS2: Transfer",       add_cs2_transfer_page),
            ("CS2: Cost",           add_cost_comparison_page),
            ("CS1: NAS aggregate",  add_cs1_results_page),
        ]
        for label, fn in pages:
            n += 1
            print(f"  [{n}] {label}")
            fn(pdf)

        # CS1 figures from paper
        fig_pages = [
            (FIG_DIR / "fig_nas_transfer_heatmap.pdf",
             "Cross-Dataset Transfer Heatmap",
             "Accuracy delta (Symbolic \u2212 Random) per source\u2192target pair. "
             "All cells positive = 100% win rate. Stars: statistical significance.",
             "CS1: Heatmap"),
            (FIG_DIR / "fig_nas_aggregate.pdf",
             "Aggregate Results: 70 Transfer Pairs",
             "Symbolic Executor achieves 100% win rate vs Random Search (89% significant) "
             "across NAS-Bench-Graph (64 pairs) and NAS-Bench-201 (6 pairs).",
             "CS1: Aggregate"),
            (FIG_DIR / "fig_kg_dominant_paths.pdf",
             "Dominant Traversal Paths (NAS GEAKG, Cora)",
             "Top-5 most-traversed paths through the NAS GEAKG. Each box is a role, colored by category. "
             "Bar length = traversal frequency. All paths follow the learned category pipeline: "
             "Topology \u2192 Activation \u2192 Training \u2192 Regularization (\u2192 Evaluation). "
             "These paths are the procedural knowledge patterns the system has learned.",
             "CS1: Dominant Paths"),
        ]
        for path, title, caption, label in fig_pages:
            n += 1
            print(f"  [{n}] {label}")
            add_figure_page(pdf, path, title, caption,
                            banner_label="CASE STUDY 1 \u2014 Neural Architecture Search",
                            banner_color=C_CS1, banner_bg=C_CS1_BG)

        n += 1
        print(f"  [{n}] Conclusions")
        add_conclusions_page(pdf)

    print(f"\n  Output: {OUT_PDF}  ({n} pages)")
    print("  Done!")


if __name__ == "__main__":
    main()
