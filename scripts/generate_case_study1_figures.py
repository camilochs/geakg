#!/usr/bin/env python3
"""Generate publication-quality figures for Case Study 1 (NAS) of the GEAKG paper.

Produces 3 PDF figures from the NAS-Bench-Graph and NAS-Bench-201 symbolic
transfer experiments:

  1. Transfer Heatmap  — delta accuracy (Symbolic - Random) per pair
  2. Variance Comparison — std comparison across methods for selected pairs
  3. Aggregate Summary   — win/significance rates for Graph vs 201

Data:
  - results/nas_bench_graph/nas_symbolic_{source}_*.json
  - results/nas_bench/nas_bench_symbolic_{source}_*.json

Output:
  - paper/figures/fig_nas_transfer_heatmap.pdf
  - paper/figures/fig_nas_variance.pdf
  - paper/figures/fig_nas_aggregate.pdf

Usage:
    uv run python scripts/generate_case_study1_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

VERDE_CENTRAL = "#0CF574"      # Spring Green — Symbolic Executor / GEAKG / "learned"
VERDE_OSCURO = "#587291"       # Blue Slate — NAS-Bench-Graph benchmark
SALVIA_CLARA = "#15E6CD"       # Turquoise — accent/highlight badges
AZUL_GRISACEO = "#2F97C1"     # Blue Green — RegEvo method
CIRUELA_APAGADA = "#1CCAD8"   # Strong Cyan — accent (heatmap negative)
GRIS_TINTA = "#2B2F36"        # Neutral dark — Random method / text
AZUL_201 = "#6366F1"          # Indigo — NAS-Bench-201 benchmark (distinct from RegEvo)

# Semantic mapping:
#   Symbolic/GEAKG  -> VERDE_CENTRAL   (green)
#   RegEvo          -> AZUL_GRISACEO   (blue)
#   Random          -> GRIS_TINTA      (dark gray)
#   NAS-Bench-Graph -> VERDE_OSCURO    (slate)
#   NAS-Bench-201   -> AZUL_201        (indigo, distinct from RegEvo)
#   Heatmap cmap    -> SALVIA_CLARA -> VERDE_OSCURO
#   Accent/highlight-> SALVIA_CLARA / CIRUELA_APAGADA

# Build a custom diverging colormap for heatmaps: red-ish to white to green
_HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "geakg_diverging",
    [CIRUELA_APAGADA, "#FFFFFF", SALVIA_CLARA, VERDE_CENTRAL, VERDE_OSCURO],
    N=256,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "results" / "nas_bench_graph"
BENCH201_DIR = ROOT / "results" / "nas_bench"
FIGURES_DIR = ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_SOURCES_ORDERED = [
    "cora", "citeseer", "pubmed", "cs", "physics", "photo", "computers", "arxiv",
]

GRAPH_ALL_DATASETS = GRAPH_SOURCES_ORDERED + ["proteins"]

BENCH201_SOURCES_ORDERED = ["cifar10", "cifar100", "ImageNet16-120"]

# Display names
DISPLAY_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "cs": "CS",
    "physics": "Physics",
    "photo": "Photo",
    "computers": "Computers",
    "arxiv": "arXiv",
    "proteins": "Proteins",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "ImageNet16-120": "ImageNet16-120",
}

METHOD_NAMES = {
    "symbolic_executor": "Symbolic",
    "reg_evolution": "RegEvo",
    "aco_cold_start": "ACO Cold",
    "random_search": "Random",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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


def _set_conditional_annotation_colors(
    ax: plt.Axes,
    values: np.ndarray,
    mask: np.ndarray,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
) -> None:
    """Set white text on dark heatmap cells and dark text on light cells."""
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
        r, g, b, _ = cmap(norm(values[row, col]))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        use_white = luminance < 0.52
        txt.set_color("white" if use_white else GRIS_TINTA)
        # Thin stroke improves readability after PDF scaling.
        txt.set_path_effects([
            mpe.withStroke(
                linewidth=1.1,
                foreground=GRIS_TINTA if use_white else "white",
                alpha=0.38 if use_white else 0.28,
            )
        ])


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _extract_source_and_timestamp(filename: str, prefix: str) -> tuple[str, str] | None:
    """Extract source dataset name and timestamp from a filename.

    For Graph:  nas_symbolic_{source}_{timestamp}.json
    For 201:    nas_bench_symbolic_{source}_{timestamp}.json
    """
    base = filename.replace(".json", "")
    # Remove the prefix
    rest = base[len(prefix):]
    # The timestamp is the last two underscore-separated parts (date_time)
    parts = rest.rsplit("_", 2)
    if len(parts) < 3:
        return None
    source = parts[0]
    timestamp = parts[1] + "_" + parts[2]
    return source, timestamp


def _pick_best_file(file_list: list[tuple[str, Path]]) -> tuple[Path, dict]:
    """Pick the best result file using priority: gpt-5.2 > most targets > latest.

    Selection criteria (in order):
      1. Prefer gpt-5.2 snapshots over other LLMs.
      2. Among same-LLM files, prefer more targets (excluding self-transfer).
      3. Among ties, prefer the latest timestamp.
    Skips files with invalid JSON.
    """
    candidates: list[tuple[int, int, str, Path, dict]] = []
    for timestamp, fp in file_list:
        try:
            with open(fp) as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        llm = loaded.get("snapshot", {}).get("llm", "unknown")
        is_gpt52 = 1 if llm == "gpt-5.2" else 0
        # Count valid targets (exclude self-transfers)
        source = loaded.get("source_dataset", "")
        targets = loaded.get("targets", {})
        valid_targets = {
            k: v for k, v in targets.items()
            if k != source and "symbolic_executor" in v.get("methods", {})
        }
        n_targets = len(valid_targets)
        candidates.append((is_gpt52, n_targets, timestamp, fp, loaded))

    if not candidates:
        raise RuntimeError(
            f"No valid JSON files found among: "
            f"{[fp.name for _, fp in file_list]}"
        )

    # Sort by (is_gpt52 DESC, n_targets DESC, timestamp DESC)
    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    best = candidates[0]
    return best[3], best[4]


def load_latest_graph_results() -> dict[str, dict]:
    """Load the best NAS-Bench-Graph symbolic result per source.

    Selects the file with the most targets; among ties, the latest timestamp.

    Returns:
        {source_dataset: parsed_json_dict}
    """
    prefix = "nas_symbolic_"
    files_by_source: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for fp in sorted(GRAPH_DIR.glob("nas_symbolic_*.json")):
        result = _extract_source_and_timestamp(fp.name, prefix)
        if result is None:
            continue
        source, timestamp = result
        files_by_source[source].append((timestamp, fp))

    data: dict[str, dict] = {}
    for source, file_list in files_by_source.items():
        if source == "proteins":
            continue  # Exclude Proteins as source
        best_fp, best_data = _pick_best_file(file_list)
        n_tgt = len(best_data.get("targets", {}))
        data[source] = best_data
        print(f"  Graph [{source}]: {best_fp.name} ({n_tgt} targets)")

    return data


def load_latest_201_results() -> dict[str, dict]:
    """Load the best NAS-Bench-201 symbolic result per source.

    Selects the file with the most targets; among ties, the latest timestamp.

    Returns:
        {source_dataset: parsed_json_dict}
    """
    prefix = "nas_bench_symbolic_"
    files_by_source: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for fp in sorted(BENCH201_DIR.glob("nas_bench_symbolic_*.json")):
        result = _extract_source_and_timestamp(fp.name, prefix)
        if result is None:
            continue
        source, timestamp = result
        files_by_source[source].append((timestamp, fp))

    data: dict[str, dict] = {}
    for source, file_list in files_by_source.items():
        best_fp, best_data = _pick_best_file(file_list)
        n_tgt = len(best_data.get("targets", {}))
        data[source] = best_data
        print(f"  201 [{source}]: {best_fp.name} ({n_tgt} targets)")

    return data


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def get_pair_stats(
    data: dict, source: str, target: str
) -> dict | None:
    """Extract method stats for a (source, target) transfer pair.

    Returns dict with keys: sym_mean, sym_std, reg_mean, reg_std,
    rand_mean, rand_std, aco_mean, aco_std, p_rand, p_reg, delta_rand, delta_reg
    """
    src_data = data.get(source)
    if src_data is None:
        return None
    targets = src_data.get("targets", {})
    tgt_data = targets.get(target)
    if tgt_data is None:
        return None
    methods = tgt_data.get("methods", {})
    sym = methods.get("symbolic_executor", {})
    reg = methods.get("reg_evolution", {})
    rand = methods.get("random_search", {})
    aco = methods.get("aco_cold_start", {})

    sym_mean = sym.get("mean_accuracy", 0.0)
    reg_mean = reg.get("mean_accuracy", 0.0)
    rand_mean = rand.get("mean_accuracy", 0.0)

    return {
        "sym_mean": sym_mean,
        "sym_std": sym.get("std_accuracy", 0.0),
        "reg_mean": reg_mean,
        "reg_std": reg.get("std_accuracy", 0.0),
        "rand_mean": rand_mean,
        "rand_std": rand.get("std_accuracy", 0.0),
        "aco_mean": aco.get("mean_accuracy", 0.0),
        "aco_std": aco.get("std_accuracy", 0.0),
        "p_rand": tgt_data.get("symbolic_vs_random_p", 1.0),
        "p_reg": tgt_data.get("symbolic_vs_regevo_p", 1.0),
        "delta_rand": sym_mean - rand_mean,
        "delta_reg": sym_mean - reg_mean,
    }


def _sig_marker(p: float) -> str:
    """Return significance marker for a p-value."""
    if p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Figure 1: Transfer Heatmap
# ---------------------------------------------------------------------------

def _heatmap_significance_note(fig: plt.Figure) -> None:
    """Add significance footnote to a heatmap figure."""
    fig.text(
        0.5, -0.01,
        r"Cell values: $\Delta$ = Symbolic $-$ Random (pp).  "
        r"Significance: * $p < 0.05$,  ** $p < 0.01$  (Wilcoxon signed-rank test).",
        ha="center", fontsize=13, style="italic", color=GRIS_TINTA,
    )


def fig1a_transfer_heatmap_graph(graph_data: dict[str, dict]) -> None:
    """NAS-Bench-Graph transfer heatmap (8 sources x 9 targets)."""

    sources = GRAPH_SOURCES_ORDERED
    all_targets_ordered = GRAPH_ALL_DATASETS

    n_src = len(sources)
    n_tgt = len(all_targets_ordered)

    delta_matrix = np.full((n_src, n_tgt), np.nan)
    annot_matrix = np.empty((n_src, n_tgt), dtype=object)

    for i, src in enumerate(sources):
        for j, tgt in enumerate(all_targets_ordered):
            if src == tgt:
                annot_matrix[i, j] = ""
                continue
            stats = get_pair_stats(graph_data, src, tgt)
            if stats is None:
                annot_matrix[i, j] = ""
                continue
            delta = stats["delta_rand"]
            p = stats["p_rand"]
            sig = _sig_marker(p)
            delta_matrix[i, j] = delta
            annot_matrix[i, j] = f"{delta:+.2f}{sig}"

    mask = np.zeros_like(delta_matrix, dtype=bool)
    for i, src in enumerate(sources):
        for j, tgt in enumerate(all_targets_ordered):
            if src == tgt:
                mask[i, j] = True

    fig, ax = plt.subplots(figsize=(14, 8))

    vmax_clip = 5.0
    display_matrix = np.clip(delta_matrix, -vmax_clip, vmax_clip)
    hm = sns.heatmap(
        display_matrix,
        ax=ax,
        mask=mask,
        annot=annot_matrix,
        fmt="",
        cmap=_HEATMAP_CMAP,
        center=0,
        vmin=-vmax_clip,
        vmax=vmax_clip,
        linewidths=1.0,
        linecolor="white",
        xticklabels=[DISPLAY_NAMES[d] for d in all_targets_ordered],
        yticklabels=[DISPLAY_NAMES[s] for s in sources],
        cbar_kws={"shrink": 0.8, "label": r"$\Delta$ Accuracy (pp, clipped at $\pm$5)"},
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    _set_conditional_annotation_colors(
        ax, display_matrix, mask, _HEATMAP_CMAP, -vmax_clip, vmax_clip
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$\Delta$ Accuracy (pp, clipped at $\pm$5)", size=14)
    ax.set_xlabel("Target Dataset", fontsize=16)
    ax.set_ylabel("Source Dataset", fontsize=16)
    ax.set_title("NAS-Bench-Graph: Cross-Dataset Transfer Heatmap", fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=40, labelsize=14)
    ax.tick_params(axis="y", rotation=0, labelsize=14)

    _heatmap_significance_note(fig)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / "fig_nas_heatmap_graph.pdf")


def fig1b_transfer_heatmap_201(bench201_data: dict[str, dict]) -> None:
    """NAS-Bench-201 transfer heatmap (3 sources x 3 targets)."""

    sources_201 = BENCH201_SOURCES_ORDERED
    all_targets_201 = BENCH201_SOURCES_ORDERED

    n_src_201 = len(sources_201)
    n_tgt_201 = len(all_targets_201)

    delta_201 = np.full((n_src_201, n_tgt_201), np.nan)
    annot_201 = np.empty((n_src_201, n_tgt_201), dtype=object)

    for i, src in enumerate(sources_201):
        for j, tgt in enumerate(all_targets_201):
            if src == tgt:
                annot_201[i, j] = ""
                continue
            stats = get_pair_stats(bench201_data, src, tgt)
            if stats is None:
                annot_201[i, j] = ""
                continue
            delta = stats["delta_rand"]
            p = stats["p_rand"]
            sig = _sig_marker(p)
            delta_201[i, j] = delta
            annot_201[i, j] = f"{delta:+.2f}{sig}"

    mask_201 = np.zeros_like(delta_201, dtype=bool)
    for i in range(n_src_201):
        mask_201[i, i] = True

    fig, ax = plt.subplots(figsize=(8, 6))

    vmax_201 = max(2.0, np.nanmax(np.abs(delta_201)))
    hm = sns.heatmap(
        delta_201,
        ax=ax,
        mask=mask_201,
        annot=annot_201,
        fmt="",
        cmap=_HEATMAP_CMAP,
        center=0,
        vmin=-vmax_201,
        vmax=vmax_201,
        linewidths=1.0,
        linecolor="white",
        xticklabels=[DISPLAY_NAMES[d] for d in all_targets_201],
        yticklabels=[DISPLAY_NAMES[s] for s in sources_201],
        cbar_kws={"shrink": 0.8, "label": r"$\Delta$ Accuracy (pp)"},
        annot_kws={"size": 16, "fontweight": "bold"},
    )
    _set_conditional_annotation_colors(
        ax, delta_201, mask_201, _HEATMAP_CMAP, -vmax_201, vmax_201
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$\Delta$ Accuracy (pp)", size=14)
    ax.set_xlabel("Target Dataset", fontsize=16)
    ax.set_ylabel("Source Dataset", fontsize=16)
    ax.set_title("NAS-Bench-201: Cross-Dataset Transfer Heatmap", fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=40, labelsize=14)
    ax.tick_params(axis="y", rotation=0, labelsize=14)

    _heatmap_significance_note(fig)
    plt.tight_layout()
    _save(fig, FIGURES_DIR / "fig_nas_heatmap_201.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Variance Comparison
# ---------------------------------------------------------------------------

def _draw_variance_panel(
    ax: plt.Axes,
    data: dict[str, dict],
    pairs: list[tuple[str, str, str]],
    title: str,
) -> None:
    """Draw a single variance-comparison panel.

    Args:
        pairs: list of (source, target, display_label)
    """
    methods = ["symbolic_executor", "reg_evolution", "random_search"]
    method_labels = ["Symbolic Executor", "RegEvo", "Random"]
    method_colors = [VERDE_CENTRAL, AZUL_GRISACEO, GRIS_TINTA]
    n_methods = len(methods)
    bar_width = 0.24
    x = np.arange(len(pairs))

    # Pre-compute std matrix for stable label offsets and y-limits.
    std_matrix = np.zeros((len(pairs), len(methods)))
    for p_idx, (src, tgt, _) in enumerate(pairs):
        stats = get_pair_stats(data, src, tgt)
        if not stats:
            continue
        std_matrix[p_idx, 0] = stats.get("sym_std", 0.0)
        std_matrix[p_idx, 1] = stats.get("reg_std", 0.0)
        std_matrix[p_idx, 2] = stats.get("rand_std", 0.0)

    max_std = float(np.max(std_matrix)) if np.max(std_matrix) > 0 else 1.0
    # Extra headroom for ratio badges + staggered labels
    y_top = max_std + max(0.65 * max_std, 0.50)
    label_fs = 8.2

    # Draw bars
    bar_offsets = []
    all_bars = []
    for m_idx, (method, label, color) in enumerate(
        zip(methods, method_labels, method_colors)
    ):
        stds = std_matrix[:, m_idx].tolist()
        offset = (m_idx - (n_methods - 1) / 2) * bar_width
        bar_offsets.append(offset)
        bars = ax.bar(
            x + offset, stds, bar_width,
            label=label, color=color, alpha=0.88,
            edgecolor="white", linewidth=0.8,
        )
        all_bars.append(bars)

    # Annotate bar values — stagger labels that would overlap
    stagger_gap = 0.045 * y_top  # minimum vertical gap between labels
    for p_idx in range(len(pairs)):
        # Collect (bar_height, method_idx) sorted by height
        entries = []
        for m_idx in range(n_methods):
            val = std_matrix[p_idx, m_idx]
            if val > 0:
                entries.append((val, m_idx))
        entries.sort(key=lambda e: e[0])

        # Compute label y-positions: start at bar_top + small pad,
        # but push up if too close to the previous label
        base_pad = 0.015 * y_top
        label_positions = []
        for rank, (val, m_idx) in enumerate(entries):
            y_candidate = val + base_pad
            if label_positions:
                prev_y = label_positions[-1]
                if y_candidate < prev_y + stagger_gap:
                    y_candidate = prev_y + stagger_gap
            label_positions.append(y_candidate)

        # Draw labels
        for (val, m_idx), y_pos in zip(entries, label_positions):
            color = method_colors[m_idx]
            bar = all_bars[m_idx][p_idx]
            x_center = bar.get_x() + bar.get_width() / 2
            txt = ax.text(
                x_center, y_pos,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=label_fs,
                color=color, fontweight="bold",
            )
            # Add subtle outline for readability
            txt.set_path_effects([
                mpe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)
            ])

    # Annotate variance-reduction ratio above each group
    ratio_font = 9.5
    for p_idx, (src, tgt, _) in enumerate(pairs):
        stats = get_pair_stats(data, src, tgt)
        if stats is None:
            continue
        sym_std = stats["sym_std"]
        rand_std = stats["rand_std"]
        worst_std = max(stats["reg_std"], rand_std)
        if worst_std > 0 and sym_std > 0:
            ratio = worst_std / sym_std
            if ratio < 1.15:
                continue
            group_max = max(sym_std, stats["reg_std"], rand_std)
            ratio_y = group_max + max(0.10, 0.16 * max_std)
            ax.text(
                p_idx, ratio_y,
                f"{ratio:.1f}\u00d7 lower",
                ha="center", va="bottom", fontsize=ratio_font,
                color=GRIS_TINTA, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor=SALVIA_CLARA,
                    edgecolor=VERDE_CENTRAL, linewidth=0.8,
                ),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=11)
    ax.set_ylabel("Std. Deviation of Accuracy (pp)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, y_top)
    sns.despine(ax=ax, left=False, bottom=False)


def fig2_variance_comparison(
    graph_data: dict[str, dict],
    bench201_data: dict[str, dict],
) -> None:
    """Two-panel variance comparison: one panel per benchmark."""

    graph_pairs = [
        ("cora", "physics", "Cora \u2192\nPhysics"),
        ("cora", "computers", "Cora \u2192\nComputers"),
        ("cora", "arxiv", "Cora \u2192\narXiv"),
        ("cora", "photo", "Cora \u2192\nPhoto"),
        ("cora", "cs", "Cora \u2192\nCS"),
    ]

    bench201_pairs = [
        ("cifar10", "cifar100", "C10 \u2192\nC100"),
        ("cifar10", "ImageNet16-120", "C10 \u2192\nIN16"),
        ("cifar100", "ImageNet16-120", "C100 \u2192\nIN16"),
        ("ImageNet16-120", "cifar100", "IN16 \u2192\nC100"),
    ]

    fig, (ax_graph, ax_201) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [5, 4]},
    )

    _draw_variance_panel(
        ax_graph, graph_data, graph_pairs,
        "(a) NAS-Bench-Graph",
    )
    _draw_variance_panel(
        ax_201, bench201_data, bench201_pairs,
        "(b) NAS-Bench-201",
    )

    # Shared legend from the left panel
    handles, labels = ax_graph.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=12.5,
        frameon=True, fancybox=True, edgecolor=SALVIA_CLARA,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.15, wspace=0.17)
    _save(fig, FIGURES_DIR / "fig_nas_variance.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Aggregate Summary
# ---------------------------------------------------------------------------

def _compute_aggregate(
    data: dict[str, dict],
    sources: list[str],
    all_datasets: list[str],
) -> dict:
    """Compute aggregate win/significance rates."""
    total_pairs = 0
    wins_rand = 0
    sig_rand = 0
    wins_reg = 0
    sig_reg = 0
    deltas_rand = []
    deltas_reg = []

    for src in sources:
        src_data = data.get(src)
        if src_data is None:
            continue
        targets = src_data.get("targets", {})
        for tgt_name, tgt_data in targets.items():
            if tgt_name == src:
                continue
            methods = tgt_data.get("methods", {})
            sym = methods.get("symbolic_executor", {})
            reg = methods.get("reg_evolution", {})
            rand = methods.get("random_search", {})

            sym_mean = sym.get("mean_accuracy")
            reg_mean = reg.get("mean_accuracy")
            rand_mean = rand.get("mean_accuracy")

            if sym_mean is None or rand_mean is None:
                continue

            total_pairs += 1
            delta_r = sym_mean - rand_mean
            deltas_rand.append(delta_r)
            if delta_r > 0:
                wins_rand += 1
            p_rand = tgt_data.get("symbolic_vs_random_p", 1.0)
            if p_rand < 0.05 and delta_r > 0:
                sig_rand += 1

            if reg_mean is not None:
                delta_re = sym_mean - reg_mean
                deltas_reg.append(delta_re)
                if delta_re > 0:
                    wins_reg += 1
                p_reg = tgt_data.get("symbolic_vs_regevo_p", 1.0)
                if p_reg < 0.05 and delta_re > 0:
                    sig_reg += 1

    return {
        "total": total_pairs,
        "wins_rand": wins_rand,
        "sig_rand": sig_rand,
        "wins_reg": wins_reg,
        "sig_reg": sig_reg,
        "win_rate_rand": 100 * wins_rand / total_pairs if total_pairs > 0 else 0,
        "sig_rate_rand": 100 * sig_rand / total_pairs if total_pairs > 0 else 0,
        "win_rate_reg": 100 * wins_reg / total_pairs if total_pairs > 0 else 0,
        "sig_rate_reg": 100 * sig_reg / total_pairs if total_pairs > 0 else 0,
        "mean_delta_rand": np.mean(deltas_rand) if deltas_rand else 0,
        "mean_delta_reg": np.mean(deltas_reg) if deltas_reg else 0,
    }


def fig3_aggregate_summary(
    graph_data: dict[str, dict],
    bench201_data: dict[str, dict],
) -> None:
    """Grouped bar chart comparing aggregate metrics: Graph vs 201."""

    # Compute aggregates
    # For Graph: exclude proteins from targets (handled in _compute_aggregate)
    graph_agg = _compute_aggregate(
        graph_data, GRAPH_SOURCES_ORDERED, GRAPH_ALL_DATASETS
    )
    bench201_agg = _compute_aggregate(
        bench201_data, BENCH201_SOURCES_ORDERED, BENCH201_SOURCES_ORDERED
    )

    print("\n  Aggregate summary (verification):")
    print(f"    Graph: {graph_agg['total']} pairs, "
          f"win_rand={graph_agg['wins_rand']}/{graph_agg['total']} "
          f"({graph_agg['win_rate_rand']:.0f}%), "
          f"sig_rand={graph_agg['sig_rand']}/{graph_agg['total']} "
          f"({graph_agg['sig_rate_rand']:.0f}%), "
          f"win_reg={graph_agg['wins_reg']}/{graph_agg['total']} "
          f"({graph_agg['win_rate_reg']:.0f}%), "
          f"sig_reg={graph_agg['sig_reg']}/{graph_agg['total']} "
          f"({graph_agg['sig_rate_reg']:.0f}%)")
    print(f"    201:   {bench201_agg['total']} pairs, "
          f"win_rand={bench201_agg['wins_rand']}/{bench201_agg['total']} "
          f"({bench201_agg['win_rate_rand']:.0f}%), "
          f"sig_rand={bench201_agg['sig_rand']}/{bench201_agg['total']} "
          f"({bench201_agg['sig_rate_rand']:.0f}%), "
          f"win_reg={bench201_agg['wins_reg']}/{bench201_agg['total']} "
          f"({bench201_agg['win_rate_reg']:.0f}%), "
          f"sig_reg={bench201_agg['sig_reg']}/{bench201_agg['total']} "
          f"({bench201_agg['sig_rate_reg']:.0f}%)")
    print(f"    Mean delta rand: Graph={graph_agg['mean_delta_rand']:.2f} pp, "
          f"201={bench201_agg['mean_delta_rand']:.2f} pp")

    # Build figure
    metrics = [
        "Win Rate\nvs Random",
        "Significance\nvs Random",
        "Win Rate\nvs RegEvo",
        "Significance\nvs RegEvo",
    ]
    graph_vals = [
        graph_agg["win_rate_rand"],
        graph_agg["sig_rate_rand"],
        graph_agg["win_rate_reg"],
        graph_agg["sig_rate_reg"],
    ]
    bench201_vals = [
        bench201_agg["win_rate_rand"],
        bench201_agg["sig_rate_rand"],
        bench201_agg["win_rate_reg"],
        bench201_agg["sig_rate_reg"],
    ]
    # Count annotations (e.g., "56/56", "5/6")
    graph_counts = [
        f"{graph_agg['wins_rand']}/{graph_agg['total']}",
        f"{graph_agg['sig_rand']}/{graph_agg['total']}",
        f"{graph_agg['wins_reg']}/{graph_agg['total']}",
        f"{graph_agg['sig_reg']}/{graph_agg['total']}",
    ]
    bench201_counts = [
        f"{bench201_agg['wins_rand']}/{bench201_agg['total']}",
        f"{bench201_agg['sig_rand']}/{bench201_agg['total']}",
        f"{bench201_agg['wins_reg']}/{bench201_agg['total']}",
        f"{bench201_agg['sig_reg']}/{bench201_agg['total']}",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    bar_width = 0.34
    color_graph = VERDE_OSCURO
    color_201 = AZUL_201

    bars_g = ax.bar(
        x - bar_width / 2, graph_vals, bar_width,
        label="NAS-Bench-Graph", color=color_graph, alpha=0.88,
        edgecolor="white", linewidth=0.8,
    )
    bars_2 = ax.bar(
        x + bar_width / 2, bench201_vals, bar_width,
        label="NAS-Bench-201", color=color_201, alpha=0.88,
        edgecolor="white", linewidth=0.8,
    )

    # Annotate with count strings
    for bar, count, val in zip(bars_g, graph_counts, graph_vals):
        txt = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            count,
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=color_graph,
        )
        txt.set_path_effects([
            mpe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)
        ])

    for bar, count, val in zip(bars_2, bench201_counts, bench201_vals):
        txt = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            count,
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=color_201,
        )
        txt.set_path_effects([
            mpe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)
        ])

    # Reference lines
    ax.axhline(y=50, color="#D32F2F", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        len(metrics) - 0.5, 51.5, "50% baseline",
        ha="right", va="bottom", fontsize=11, color="#D32F2F", fontstyle="italic",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=15)
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        "Aggregate Transfer Performance: Symbolic Executor vs Baselines",
        fontsize=16, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=13, loc="upper center", ncol=2, frameon=True,
              fancybox=True, edgecolor=SALVIA_CLARA,
              bbox_to_anchor=(0.5, -0.10))
    sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()
    _save(fig, FIGURES_DIR / "fig_nas_aggregate.pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_style()

    print("Loading NAS-Bench-Graph results ...")
    graph_data = load_latest_graph_results()
    print(f"  Loaded {len(graph_data)} source datasets\n")

    print("Loading NAS-Bench-201 results ...")
    bench201_data = load_latest_201_results()
    print(f"  Loaded {len(bench201_data)} source datasets\n")

    print("Figure 1a: Transfer Heatmap (NAS-Bench-Graph) ...")
    fig1a_transfer_heatmap_graph(graph_data)

    print("\nFigure 1b: Transfer Heatmap (NAS-Bench-201) ...")
    fig1b_transfer_heatmap_201(bench201_data)

    print("\nFigure 2: Variance Comparison ...")
    fig2_variance_comparison(graph_data, bench201_data)

    print("\nFigure 3: Aggregate Summary ...")
    fig3_aggregate_summary(graph_data, bench201_data)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
