#!/usr/bin/env python3
"""Generate KG Analysis figures for Section 6 of the paper.

Produces three publication-quality figures from NAS benchmark data:
  1. fig_kg_pheromone_heatmap.pdf — 18x18 learned pheromone matrix
  2. fig_kg_pheromone_entropy.pdf — Entropy comparison across topologies
  3. fig_kg_dominant_paths.pdf — Top-5 dominant paths with annotations

Data sources:
  - results/nas_bench_graph/nasbench_graph_pheromone_*.json
  - results/nas_bench_graph/nasbench_graph_llm_sweep_*.json

Output: paper/figures/

Usage:
    uv run python scripts/generate_kg_analysis_figures.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

VERDE_CENTRAL = "#0CF574"      # Spring Green — Symbolic Executor / GEAKG / "learned"
VERDE_OSCURO = "#587291"       # Blue Slate — NAS-Bench-Graph / Regularization category
SALVIA_CLARA = "#15E6CD"       # Turquoise — accent/highlight / Evaluation category
AZUL_GRISACEO = "#2F97C1"     # Blue Green — RegEvo / Activation category
CIRUELA_APAGADA = "#1CCAD8"   # Strong Cyan — Training category
GRIS_TINTA = "#2B2F36"        # Neutral dark — Random / text
GRIS_NEUTRO = "#94A3B8"       # Slate gray — neutral "before" / Uniform state

# Sequential green colormap for pheromone heatmap: Salvia clara -> Verde oscuro
_PHEROMONE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "geakg_sequential",
    ["#FFFFFF", SALVIA_CLARA, VERDE_CENTRAL, VERDE_OSCURO],
    N=256,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAS_CATEGORIES = ["topology", "activation", "training", "regularization", "evaluation"]

# Category palette using the project colors
CATEGORY_PALETTE = {
    "topology": VERDE_CENTRAL,
    "activation": AZUL_GRISACEO,
    "training": CIRUELA_APAGADA,
    "regularization": VERDE_OSCURO,
    "evaluation": SALVIA_CLARA,
}

ROLES_ORDERED = [
    "topo_feedforward", "topo_residual", "topo_recursive", "topo_cell_based",
    "act_standard", "act_modern", "act_parametric", "act_mixed",
    "train_optimizer", "train_schedule", "train_augmentation", "train_loss",
    "reg_dropout", "reg_normalization", "reg_weight_decay", "reg_structural",
    "eval_proxy", "eval_full",
]

ROLE_TO_CATEGORY = {
    "topo_feedforward": "topology", "topo_residual": "topology",
    "topo_recursive": "topology", "topo_cell_based": "topology",
    "act_standard": "activation", "act_modern": "activation",
    "act_parametric": "activation", "act_mixed": "activation",
    "train_optimizer": "training", "train_schedule": "training",
    "train_augmentation": "training", "train_loss": "training",
    "reg_dropout": "regularization", "reg_normalization": "regularization",
    "reg_weight_decay": "regularization", "reg_structural": "regularization",
    "eval_proxy": "evaluation", "eval_full": "evaluation",
}

OPTIM_ROLES = [
    "CONST_GREEDY", "CONST_INSERTION", "CONST_SAVINGS", "CONST_RANDOM",
    "LS_INTENSIFY_SMALL", "LS_INTENSIFY_MEDIUM", "LS_INTENSIFY_LARGE",
    "LS_CHAIN",
    "PERT_ESCAPE_SMALL", "PERT_ESCAPE_LARGE", "PERT_ADAPTIVE",
]

OPTIM_CATEGORIES = ["construction", "local_search", "perturbation"]

OPTIM_ROLE_TO_CAT = {
    "CONST_GREEDY": "construction", "CONST_INSERTION": "construction",
    "CONST_SAVINGS": "construction", "CONST_RANDOM": "construction",
    "LS_INTENSIFY_SMALL": "local_search", "LS_INTENSIFY_MEDIUM": "local_search",
    "LS_INTENSIFY_LARGE": "local_search", "LS_CHAIN": "local_search",
    "PERT_ESCAPE_SMALL": "perturbation", "PERT_ESCAPE_LARGE": "perturbation",
    "PERT_ADAPTIVE": "perturbation",
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "axes.edgecolor": GRIS_TINTA,
        "text.color": GRIS_TINTA,
        "axes.labelcolor": GRIS_TINTA,
        "xtick.color": GRIS_TINTA,
        "ytick.color": GRIS_TINTA,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#E0E0E0",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


def _set_conditional_annotation_colors(
    ax: plt.Axes,
    values: np.ndarray,
    mask: np.ndarray,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
) -> None:
    """Use white annotation text on dark heatmap cells."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    n_rows, n_cols = values.shape
    for txt in ax.texts:
        label = txt.get_text().strip()
        if not label:
            continue
        x, y = txt.get_position()
        col = int(round(x - 0.5))
        row = int(round(y - 0.5))
        if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
            continue
        if mask[row, col] or np.isnan(values[row, col]):
            continue
        val = values[row, col]
        r, g, b, _ = cmap(norm(val))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Force white on high-intensity/darker cells and on low-luminance colors.
        use_white = (val >= 0.62) or (luminance < 0.60)
        txt.set_color("white" if use_white else GRIS_TINTA)
        txt.set_fontweight("bold")
        txt.set_path_effects([
            mpe.withStroke(
                linewidth=0.9,
                foreground=(GRIS_TINTA if use_white else "white"),
                alpha=0.95,
            )
        ])


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_pheromone_data(results_dir: Path) -> dict:
    """Load pheromone analysis files."""
    data = {}
    for fp in sorted(results_dir.glob("nasbench_graph_pheromone_*.json")):
        with open(fp) as f:
            d = json.load(f)
        ds = d.get("dataset", "unknown")
        data[ds] = d
    return data


def load_llm_sweep_data(results_dir: Path) -> dict:
    """Load llm_sweep files for multi-topology analysis."""
    all_data = {}
    for fp in sorted(results_dir.glob("nasbench_graph_llm_sweep_*.json")):
        with open(fp) as f:
            d = json.load(f)
        llm = d.get("llm", "none")
        topo = llm if llm in ("gpt-5.2", "gpt-4o-mini") else "hardcoded"
        ds = d.get("dataset", "unknown")
        key = (topo, ds)
        if key not in all_data:
            all_data[key] = d
    return all_data


# ---------------------------------------------------------------------------
# Figure 1: Pheromone Heatmap (18x18)
# ---------------------------------------------------------------------------

def fig_pheromone_heatmap(pheromone_data: dict, out_dir: Path) -> None:
    """18x18 pheromone heatmap from the learned GEAKG."""
    # Use cora as primary dataset (most data)
    if "cora" not in pheromone_data:
        ds = next(iter(pheromone_data))
    else:
        ds = "cora"

    d = pheromone_data[ds]
    pheromones = d["pheromones"]

    n = len(ROLES_ORDERED)
    role_idx = {r: i for i, r in enumerate(ROLES_ORDERED)}
    mat = np.zeros((n, n))

    for key, val in pheromones.items():
        parts = key.split("->")
        if len(parts) == 2 and parts[0] in role_idx and parts[1] in role_idx:
            mat[role_idx[parts[0]], role_idx[parts[1]]] = val

    short_labels = []
    for r in ROLES_ORDERED:
        parts = r.split("_", 1)
        short_labels.append(parts[1] if len(parts) > 1 else r)

    fig, ax = plt.subplots(figsize=(11, 11))

    mask = mat == 0
    hm = sns.heatmap(
        mat, mask=mask, ax=ax,
        xticklabels=short_labels, yticklabels=short_labels,
        cmap=_PHEROMONE_CMAP, linewidths=0.5, linecolor="white",
        annot=np.where(mat > 0.01, np.round(mat, 2), np.nan),
        fmt=".2f", annot_kws={"size": 11.5, "fontweight": "bold"},
        cbar=False,
        vmin=0, vmax=1.0,
    )
    _set_conditional_annotation_colors(ax, mat, mask, _PHEROMONE_CMAP, 0, 1.0)

    # Category separators
    boundaries = [0, 4, 8, 12, 16, 18]
    for b in boundaries[1:-1]:
        ax.axhline(b, color=GRIS_TINTA, linewidth=1.5)
        ax.axvline(b, color=GRIS_TINTA, linewidth=1.5)

    # Disable grid behind the heatmap
    ax.grid(False)

    ax.set_title(
        f"Learned Pheromone Matrix $\\Phi$ (NAS GEAKG, {ds.capitalize()} dataset)",
        fontsize=16, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Target Role $r_j$", fontsize=15, labelpad=10)
    ax.set_ylabel("Source Role $r_i$", fontsize=15)
    ax.tick_params(axis="x", labelsize=11, rotation=45)
    ax.tick_params(axis="y", labelsize=11)

    # Room for right labels and bottom colorbar — extra bottom space to avoid overlap
    fig.subplots_adjust(right=0.82, bottom=0.22)

    # Horizontal colorbar at the bottom, well below x-axis labels
    import matplotlib.cm as mcm
    sm = plt.cm.ScalarMappable(cmap=_PHEROMONE_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.04, 0.55, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Learned Pheromone Weight $\\tau_{ij}$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # Category labels on right side with bracket lines
    cat_names = ["Topology", "Activation", "Training", "Regularization", "Evaluation"]
    cat_ranges = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 18)]
    for (y0, y1), cat_name, cat_key in zip(cat_ranges, cat_names, NAS_CATEGORIES):
        mid = (y0 + y1) / 2
        color = CATEGORY_PALETTE[cat_key]
        # Bracket: short horizontal lines at top/bottom, vertical line connecting
        bx = n + 0.3  # x position for bracket start
        ax.plot([bx, bx + 0.3], [y0 + 0.1, y0 + 0.1], color=color, lw=1.2,
                clip_on=False, solid_capstyle="round")
        ax.plot([bx, bx + 0.3], [y1 - 0.1, y1 - 0.1], color=color, lw=1.2,
                clip_on=False, solid_capstyle="round")
        ax.plot([bx + 0.3, bx + 0.3], [y0 + 0.1, y1 - 0.1], color=color, lw=1.2,
                clip_on=False, solid_capstyle="round")
        # Label
        ax.text(bx + 0.5, mid, cat_name, va="center", ha="left", fontsize=11,
                fontweight="bold", color=color, clip_on=False)

    _save(fig, out_dir / "fig_kg_pheromone_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Pheromone Entropy Comparison
# ---------------------------------------------------------------------------

def fig_pheromone_entropy(llm_sweep_data: dict, out_dir: Path) -> None:
    """Violin plot of pheromone entropy by topology, showing knowledge refinement."""
    import pandas as pd

    TOPOLOGY_PALETTE = {
        "hardcoded": VERDE_OSCURO,        # Slate — human-designed baseline
        "gpt-5.2": VERDE_CENTRAL,         # Green — best LLM
        "gpt-4o-mini": "#F97316",         # Orange — mid-tier LLM (high contrast)
    }
    TOPOLOGY_LABELS = {
        "hardcoded": "Hardcoded",
        "gpt-5.2": "GPT-5.2",
        "gpt-4o-mini": "GPT-4o-mini",
    }

    records = []
    for (topo, ds), entry in llm_sweep_data.items():
        runs = entry.get("runs", [])
        for run in runs:
            ph = run.get("pheromones_display", {})
            if not ph:
                continue
            vals = list(ph.values())
            total = sum(vals)
            if total > 0:
                p = np.array(vals) / total
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p))
                active_edges = int(np.sum(np.array(vals) > 0))
            else:
                entropy = 0.0
                active_edges = 0
            records.append({
                "Topology": TOPOLOGY_LABELS.get(topo, topo),
                "Entropy (bits)": entropy,
                "Dataset": ds,
                "Active edges": active_edges,
            })

    if not records:
        print("  [skip] entropy: no data")
        return

    df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel A: Violin plot
    order = [v for v in ["Hardcoded", "GPT-5.2", "GPT-4o-mini"] if v in df["Topology"].unique()]
    palette = {TOPOLOGY_LABELS[t]: TOPOLOGY_PALETTE[t] for t in TOPOLOGY_PALETTE
               if TOPOLOGY_LABELS[t] in df["Topology"].unique()}

    if order:
        sns.violinplot(
            data=df, x="Topology", y="Entropy (bits)", hue="Topology",
            order=order, palette=palette, inner="box", linewidth=1.2,
            legend=False, ax=ax1,
        )
        sns.stripplot(
            data=df, x="Topology", y="Entropy (bits)", order=order,
            color=GRIS_TINTA, alpha=0.2, size=3, jitter=True, ax=ax1,
        )

    ax1.set_title("(a) Pheromone Entropy by L0 Topology", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Shannon Entropy (bits)", fontsize=12)
    ax1.set_xlabel("")
    sns.despine(ax=ax1, left=False, bottom=False)

    # Panel B: Uniform vs Learned entropy comparison
    # Keep one reference topology to avoid mixing different edge supports.
    ref_topology = "Hardcoded" if "Hardcoded" in df["Topology"].unique() else order[0]
    ref_df = df[df["Topology"] == ref_topology]
    if ref_df.empty:
        ref_df = df

    n_edges_typical = int(round(np.median(ref_df["Active edges"])))
    n_edges_typical = max(n_edges_typical, 2)
    uniform_entropy = np.log2(n_edges_typical)

    learned_entropies = ref_df["Entropy (bits)"].values
    mean_learned = np.mean(learned_entropies)

    labels = ["Uniform\n(no learning)", "Learned\n(after ACO)"]
    values = [uniform_entropy, mean_learned]
    bars = ax2.bar(
        labels,
        values,
        color=[GRIS_NEUTRO, VERDE_CENTRAL],
        edgecolor=[GRIS_TINTA, VERDE_OSCURO],
        linewidth=1.6,
        width=0.52,
        hatch=["////", "..."],
        zorder=3,
    )
    # Emphasize the bar tops to make small differences visible at print scale.
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax2.scatter(x, y, s=58, color="white", edgecolor=GRIS_TINTA, linewidth=1.1, zorder=4)
    ax2.axhline(
        uniform_entropy, color=GRIS_NEUTRO, linestyle="--", alpha=0.65, linewidth=1.4,
        label="Uniform reference"
    )
    delta_bits = uniform_entropy - mean_learned
    delta_pct = (delta_bits / uniform_entropy) * 100 if uniform_entropy > 0 else 0.0

    # Value labels on bars
    for bar, val in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        ax2.text(
            x, val + 0.02, f"{val:.3f}",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=GRIS_TINTA, zorder=5
        )

    # Zoom y-range to emphasize small differences while keeping labels explicit.
    span = abs(delta_bits)
    pad = max(0.04, span * 0.9)
    y_min = min(uniform_entropy, mean_learned) - (1.1 * pad)
    y_max = max(uniform_entropy, mean_learned) + (1.8 * pad)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(axis="y", alpha=0.22, linewidth=0.9)
    ax2.set_axisbelow(True)

    # Bracket + delta annotation between bars
    x0 = bars[0].get_x() + bars[0].get_width() / 2
    x1 = bars[1].get_x() + bars[1].get_width() / 2
    top = max(uniform_entropy, mean_learned) + (0.95 * pad)
    ax2.plot(
        [x0, x0, x1, x1],
        [uniform_entropy + 0.18 * pad, top, top, mean_learned + 0.18 * pad],
        color=GRIS_TINTA, linewidth=1.6, zorder=4
    )
    if delta_bits >= 0:
        delta_txt = f"{delta_pct:.1f}% reduction  (Δ={delta_bits:.3f} bits)"
        delta_color = VERDE_OSCURO
    else:
        delta_txt = f"{abs(delta_pct):.1f}% increase  (Δ={abs(delta_bits):.3f} bits)"
        delta_color = CIRUELA_APAGADA
    ax2.text(
        (x0 + x1) / 2, top + 0.10 * pad, delta_txt,
        ha="center", va="bottom", fontsize=12, fontweight="bold", color=delta_color, zorder=5
    )
    # Direct vertical difference cue beside the bars.
    x_arrow = x1 + 0.22
    y_start, y_end = sorted([uniform_entropy, mean_learned])
    ax2.annotate(
        "",
        xy=(x_arrow, y_end),
        xytext=(x_arrow, y_start),
        arrowprops=dict(arrowstyle="<->", lw=1.6, color=delta_color),
        zorder=5,
    )
    ax2.text(
        x_arrow + 0.03, (y_start + y_end) / 2, f"Δ {abs(delta_bits):.3f} bits",
        ha="left", va="center", fontsize=11, color=delta_color, fontweight="bold", zorder=5
    )
    ax2.text(
        0.98, 0.02, "zoomed y-axis", transform=ax2.transAxes,
        ha="right", va="bottom", fontsize=10, color=GRIS_TINTA, style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax2.text(
        0.02, 0.02, f"reference: {ref_topology}",
        transform=ax2.transAxes, ha="left", va="bottom",
        fontsize=10, color=GRIS_TINTA, style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax2.set_title("(b) Knowledge Refinement", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Shannon Entropy (bits)", fontsize=12)
    sns.despine(ax=ax2, left=False, bottom=False)

    plt.tight_layout()
    _save(fig, out_dir / "fig_kg_pheromone_entropy.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Dominant Paths
# ---------------------------------------------------------------------------

def fig_dominant_paths(pheromone_data: dict, out_dir: Path) -> None:
    """Top-5 dominant paths with category-colored boxes."""
    if "cora" not in pheromone_data:
        ds = next(iter(pheromone_data))
    else:
        ds = "cora"

    d = pheromone_data[ds]
    top_paths = d.get("top_paths", [])

    if not top_paths:
        print("  [skip] dominant paths: no data")
        return

    top5 = top_paths[:5]
    max_count = top5[0]["count"]
    max_len = max(len(p["path"]) for p in top5)

    fig, ax = plt.subplots(figsize=(max(16, max_len * 1.8), len(top5) * 1.3 + 2.0))

    box_w = 1.5
    box_h = 0.55
    y_gap = 0.95
    x_gap = 0.2

    for row, pinfo in enumerate(top5):
        path = pinfo["path"]
        count = pinfo["count"]
        y = (len(top5) - 1 - row) * y_gap

        for col, role in enumerate(path):
            x = col * (box_w + x_gap) + 1.0
            cat = ROLE_TO_CATEGORY.get(role, "topology")
            color = CATEGORY_PALETTE[cat]

            short = role.split("_", 1)[1] if "_" in role else role

            fancy = mpatches.FancyBboxPatch(
                (x, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.88,
            )
            ax.add_patch(fancy)
            # Use dark text for light backgrounds, white for dark backgrounds
            text_color = "white" if cat in ("regularization",) else GRIS_TINTA
            ax.text(x + box_w / 2, y, short,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=text_color)

            if col < len(path) - 1:
                ax.annotate(
                    "", xy=(x + box_w + x_gap * 0.15, y),
                    xytext=(x + box_w - 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS_TINTA, lw=1.4),
                )

        # Frequency bar
        bar_x = (max_len) * (box_w + x_gap) + 1.5
        bar_w = 2.5 * count / max_count
        ax.barh(y, bar_w, height=box_h * 0.6, left=bar_x,
                color=VERDE_CENTRAL, alpha=0.8, edgecolor=VERDE_OSCURO, linewidth=0.8)
        ax.text(bar_x + bar_w + 0.15, y, f"n={count}",
                ha="left", va="center", fontsize=11, fontweight="bold", color=GRIS_TINTA)

    # Rank labels
    for row in range(len(top5)):
        y = (len(top5) - 1 - row) * y_gap
        ax.text(0.5, y, f"#{row + 1}", ha="right", va="center", fontsize=12,
                fontweight="bold", color=GRIS_TINTA)

    ax.set_xlim(-0.2, (max_len) * (box_w + x_gap) + 5)
    ax.set_ylim(-0.8, (len(top5) - 0.5) * y_gap + 0.8)
    ax.axis("off")

    # Category legend
    handles = [mpatches.Patch(color=CATEGORY_PALETTE[c], label=c.capitalize())
               for c in NAS_CATEGORIES]
    ax.legend(handles=handles, loc="lower right", fontsize=11, ncol=5,
              frameon=True, fancybox=True, edgecolor=SALVIA_CLARA)

    ax.set_title(
        f"Dominant Traversal Paths in the NAS GEAKG ({ds.capitalize()} dataset)",
        fontsize=15, fontweight="bold", pad=15,
    )

    _save(fig, out_dir / "fig_kg_dominant_paths.pdf")


# ---------------------------------------------------------------------------
# Cross-Dataset Pearson Correlation Analysis (W2a)
# ---------------------------------------------------------------------------

def cross_dataset_pearson(llm_sweep_data: dict) -> dict:
    """Compute pairwise Pearson correlations of pheromone vectors across datasets.

    For the hardcoded (no-LLM) topology, extracts the mean pheromone vector
    per dataset (averaged over 30 runs) and computes pairwise correlations.

    Returns a dict with min/mean/max correlation, p-values, and per-pair details.
    """
    # Filter to hardcoded topology only
    topo = "hardcoded"
    datasets = sorted([ds for (t, ds) in llm_sweep_data if t == topo])

    if len(datasets) < 2:
        print(f"  [skip] cross-dataset Pearson: only {len(datasets)} datasets for {topo}")
        return {}

    # Build common edge set
    all_edge_keys: set[str] = set()
    for ds in datasets:
        entry = llm_sweep_data[(topo, ds)]
        for run in entry["runs"]:
            all_edge_keys.update(run.get("pheromones_display", {}).keys())
    edge_list = sorted(all_edge_keys)

    # Build mean pheromone vector per dataset
    ds_vectors: dict[str, np.ndarray] = {}
    for ds in datasets:
        entry = llm_sweep_data[(topo, ds)]
        vec = np.zeros(len(edge_list))
        n = 0
        for run in entry["runs"]:
            ph = run.get("pheromones_display", {})
            for i, e in enumerate(edge_list):
                vec[i] += ph.get(e, 0.0)
            n += 1
        if n > 0:
            vec /= n
        ds_vectors[ds] = vec

    # Pairwise Pearson correlations
    pairs = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            r, p = sp_stats.pearsonr(ds_vectors[datasets[i]], ds_vectors[datasets[j]])
            pairs.append({
                "ds_a": datasets[i],
                "ds_b": datasets[j],
                "r": r,
                "p": p,
            })

    rs = [p["r"] for p in pairs]
    ps = [p["p"] for p in pairs]

    result = {
        "topology": topo,
        "n_datasets": len(datasets),
        "n_edges": len(edge_list),
        "n_pairs": len(pairs),
        "r_min": float(np.min(rs)),
        "r_mean": float(np.mean(rs)),
        "r_max": float(np.max(rs)),
        "r_std": float(np.std(rs)),
        "p_max": float(np.max(ps)),
        "all_significant": all(p < 0.001 for p in ps),
        "pairs": pairs,
    }

    # Print summary
    print(f"\n  Cross-Dataset Pearson Correlation ({topo} topology):")
    print(f"    Datasets: {len(datasets)}, Edges: {len(edge_list)}, Pairs: {len(pairs)}")
    print(f"    r: min={result['r_min']:.4f}, mean={result['r_mean']:.4f}, max={result['r_max']:.4f}")
    print(f"    All p < 0.001: {result['all_significant']} (max p = {result['p_max']:.2e})")

    # Print extremes
    pairs_sorted = sorted(pairs, key=lambda x: x["r"])
    print(f"    Lowest:  r={pairs_sorted[0]['r']:.4f} ({pairs_sorted[0]['ds_a']} vs {pairs_sorted[0]['ds_b']})")
    print(f"    Highest: r={pairs_sorted[-1]['r']:.4f} ({pairs_sorted[-1]['ds_a']} vs {pairs_sorted[-1]['ds_b']})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_style()

    results_dir = Path("results/nas_bench_graph")
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pheromone data ...")
    pheromone_data = load_pheromone_data(results_dir)
    print(f"  Loaded {len(pheromone_data)} datasets")

    print("Loading llm_sweep data ...")
    llm_sweep_data = load_llm_sweep_data(results_dir)
    print(f"  Loaded {len(llm_sweep_data)} (topology, dataset) entries")

    print("\nFig 1: Pheromone Heatmap ...")
    fig_pheromone_heatmap(pheromone_data, out_dir)

    print("Fig 2: Pheromone Entropy ...")
    fig_pheromone_entropy(llm_sweep_data, out_dir)

    print("Fig 3: Dominant Paths ...")
    fig_dominant_paths(pheromone_data, out_dir)

    print("\nCross-Dataset Pearson Correlation ...")
    corr_result = cross_dataset_pearson(llm_sweep_data)

    print(f"\nAll KG analysis figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
