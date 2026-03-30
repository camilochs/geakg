#!/usr/bin/env python3
"""Structural Analysis of the NAS Knowledge Graph — Figures for Paper.

Generates 6 publication-quality figures from the NAS benchmark experiments:
  1. Topology Comparison (3 MetaGraphs side-by-side)
  2. Pheromone Heatmap (learned knowledge)
  3. Entropy Comparison (ACO selectivity)
  4. Role Frequency & Category Allocation
  5. Dominant Paths Analysis
  6. Cross-Dataset Stability (transferability evidence)

Data: results/nas_bench_graph/nasbench_graph_llm_sweep_*.json
Output: results/nas_bench_graph/figures/

Usage:
    uv run python scripts/analyze_nas_structure.py
    uv run python scripts/analyze_nas_structure.py --format png --dpi 150
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Style & Constants
# ---------------------------------------------------------------------------

NAS_CATEGORIES = ["topology", "activation", "training", "regularization", "evaluation"]

CATEGORY_PALETTE = {
    "topology":       "#66c2a5",
    "activation":     "#8da0cb",
    "training":       "#fc8d62",
    "regularization": "#e78ac3",
    "evaluation":     "#a6d854",
}

TOPOLOGY_PALETTE = {
    "hardcoded":   "#7570b3",
    "gpt-5.2":    "#1b9e77",
    "gpt-4o-mini": "#d95f02",
}

TOPOLOGY_LABELS = {
    "hardcoded":   "Hardcoded",
    "gpt-5.2":    "GPT-5.2",
    "gpt-4o-mini": "GPT-4o-mini",
}

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

ROLES_ORDERED = [
    "topo_feedforward", "topo_residual", "topo_recursive", "topo_cell_based",
    "act_standard", "act_modern", "act_parametric", "act_mixed",
    "train_optimizer", "train_schedule", "train_augmentation", "train_loss",
    "reg_dropout", "reg_normalization", "reg_weight_decay", "reg_structural",
    "eval_proxy", "eval_full",
]

CATEGORY_GROUPS = {
    "Citation": ["cora", "citeseer", "pubmed"],
    "Co-author": ["cs", "physics"],
    "Amazon": ["computers", "photo"],
    "Protein": ["proteins"],
    "OGB": ["arxiv"],
}

DATASET_ORDER = ["cora", "citeseer", "pubmed", "cs", "physics",
                 "computers", "photo", "proteins", "arxiv"]


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    sns.set_palette("Set2")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _identify_topology(data: dict) -> str:
    """Determine topology name from a result file."""
    llm = data.get("llm", "none")
    if llm in ("gpt-5.2", "gpt-4o-mini"):
        return llm
    # 'none' or missing → hardcoded
    return "hardcoded"


def load_all_results(results_dir: str | Path) -> dict:
    """Load all llm_sweep JSON files.

    Returns:
        {(topology, dataset): {"meta": {...}, "runs": [...]}}
    """
    results_dir = Path(results_dir)
    all_data: dict[tuple[str, str], dict] = {}

    for fp in sorted(results_dir.glob("nasbench_graph_llm_sweep_*.json")):
        with open(fp) as f:
            data = json.load(f)

        topo = _identify_topology(data)
        dataset = data.get("dataset", "unknown")

        # Skip baseline-only files (operator_mode = base_A0) unless they are the
        # only hardcoded entries available.  We prefer the GEAKG (L0+L1) ones.
        run0 = data.get("runs", [{}])[0]
        op_mode = run0.get("operator_mode", data.get("operator_mode", ""))

        key = (topo, dataset)
        if key in all_data:
            # Keep the one with richer data (more pheromone edges)
            existing_edges = len(all_data[key]["runs"][0].get("pheromones_display", {}))
            new_edges = len(run0.get("pheromones_display", {}))
            if new_edges <= existing_edges:
                continue

        all_data[key] = {
            "meta": {
                "topology": topo,
                "dataset": dataset,
                "n_runs": data.get("n_runs", 0),
                "mean_accuracy": data.get("mean_accuracy"),
                "std_accuracy": data.get("std_accuracy"),
                "l0_stats": data.get("l0_stats", {}),
                "l1_stats": data.get("l1_stats", {}),
                "operator_mode": op_mode,
                "mode_label": data.get("mode_label", ""),
            },
            "runs": data.get("runs", []),
        }

    return all_data


# ---------------------------------------------------------------------------
# Edge Extraction
# ---------------------------------------------------------------------------

def extract_edges_from_pheromones(runs: list[dict]) -> list[tuple[str, str]]:
    """Get unique edge list from pheromones_display of first run."""
    if not runs:
        return []
    pheromones = runs[0].get("pheromones_display", {})
    edges = []
    for key in pheromones:
        parts = key.split("->")
        if len(parts) == 2:
            edges.append((parts[0], parts[1]))
    return edges


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_graph_metrics(edges: list[tuple[str, str]]) -> dict:
    """Compute structural metrics for a set of edges."""
    G = nx.DiGraph()
    G.add_nodes_from(ROLES_ORDERED)
    G.add_edges_from(edges)

    n_edges = len(edges)
    n_nodes = len(ROLES_ORDERED)
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    # Feedback loops: eval_* → non-eval
    feedback = [(s, t) for s, t in edges
                if ROLE_TO_CATEGORY.get(s) == "evaluation"
                and ROLE_TO_CATEGORY.get(t) != "evaluation"]

    # Degree stats by category
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    cat_in: dict[str, list[int]] = defaultdict(list)
    cat_out: dict[str, list[int]] = defaultdict(list)
    for role in ROLES_ORDERED:
        cat = ROLE_TO_CATEGORY[role]
        cat_in[cat].append(in_deg.get(role, 0))
        cat_out[cat].append(out_deg.get(role, 0))

    return {
        "n_edges": n_edges,
        "density": density,
        "n_feedback": len(feedback),
        "feedback_edges": feedback,
        "mean_in_by_cat": {c: np.mean(v) for c, v in cat_in.items()},
        "mean_out_by_cat": {c: np.mean(v) for c, v in cat_out.items()},
    }


def compute_pheromone_stats(runs: list[dict]) -> dict:
    """Aggregate pheromone statistics across runs."""
    all_edges: set[str] = set()
    for run in runs:
        all_edges.update(run.get("pheromones_display", {}).keys())

    edge_list = sorted(all_edges)
    n_runs = len(runs)
    matrix = np.zeros((len(edge_list), n_runs))

    for j, run in enumerate(runs):
        ph = run.get("pheromones_display", {})
        for i, e in enumerate(edge_list):
            matrix[i, j] = ph.get(e, 0.0)

    mean_per_edge = matrix.mean(axis=1)
    std_per_edge = matrix.std(axis=1)

    # Shannon entropy per run (normalise pheromones to probability)
    entropies = []
    for j in range(n_runs):
        col = matrix[:, j]
        total = col.sum()
        if total > 0:
            p = col / total
            p = p[p > 0]
            entropies.append(-np.sum(p * np.log2(p)))
        else:
            entropies.append(0.0)

    return {
        "edge_list": edge_list,
        "mean_per_edge": dict(zip(edge_list, mean_per_edge)),
        "std_per_edge": dict(zip(edge_list, std_per_edge)),
        "entropies": entropies,
        "matrix": matrix,  # edges × runs
    }


def compute_role_stats(runs: list[dict]) -> dict:
    """Aggregate role frequency across runs."""
    totals: dict[str, list[int]] = {r: [] for r in ROLES_ORDERED}
    for run in runs:
        freq = run.get("role_frequency", {})
        for r in ROLES_ORDERED:
            totals[r].append(freq.get(r, 0))

    return {
        "mean_freq": {r: np.mean(v) for r, v in totals.items()},
        "std_freq": {r: np.std(v) for r, v in totals.items()},
        "raw": totals,
    }


def compute_path_stats(runs: list[dict]) -> dict:
    """Aggregate path information across runs."""
    path_counter: Counter = Counter()
    lengths: list[int] = []

    for run in runs:
        for p in run.get("top_paths", []):
            path_tuple = tuple(p["path"])
            count = p.get("count", 1)
            path_counter[path_tuple] += count
            lengths.extend([len(path_tuple)] * count)

    return {
        "top_paths": path_counter.most_common(20),
        "mean_length": np.mean(lengths) if lengths else 0,
        "std_length": np.std(lengths) if lengths else 0,
    }


# ---------------------------------------------------------------------------
# Figure 1: Topology Comparison
# ---------------------------------------------------------------------------

def _make_layer_pos() -> dict[str, tuple[float, float]]:
    """Create fixed positions for 18 NAS roles in 5 layers."""
    pos = {}
    layer_y = {
        "topology": 4, "activation": 2, "training": 0,
        "regularization": -2, "evaluation": -4,
    }
    for cat in NAS_CATEGORIES:
        roles_in_cat = [r for r in ROLES_ORDERED if ROLE_TO_CATEGORY[r] == cat]
        n = len(roles_in_cat)
        x_start = -(n - 1) / 2 * 2.5
        for i, r in enumerate(roles_in_cat):
            pos[r] = (x_start + i * 2.5, layer_y[cat])
    return pos


def fig1_topology_comparison(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """3 MetaGraphs side-by-side with structural annotations."""
    topologies = ["hardcoded", "gpt-5.2", "gpt-4o-mini"]
    pos = _make_layer_pos()

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    for idx, topo in enumerate(topologies):
        ax = axes[idx]

        # Find any dataset for this topology to get edge list
        edges = []
        for (t, d), entry in all_data.items():
            if t == topo:
                edges = extract_edges_from_pheromones(entry["runs"])
                break

        if not edges:
            ax.text(0.5, 0.5, f"No data for {topo}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(TOPOLOGY_LABELS.get(topo, topo), fontsize=16, fontweight="bold")
            ax.axis("off")
            continue

        metrics = compute_graph_metrics(edges)

        G = nx.DiGraph()
        G.add_nodes_from(ROLES_ORDERED)
        G.add_edges_from(edges)

        # Identify feedback edges
        feedback_set = set(metrics["feedback_edges"])

        # Draw regular edges
        regular_edges = [e for e in edges if e not in feedback_set]
        nx.draw_networkx_edges(
            G, pos, edgelist=regular_edges,
            width=1.2, alpha=0.6, edge_color="#555555",
            arrows=True, arrowsize=12, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=18, min_target_margin=18,
            ax=ax,
        )

        # Draw feedback edges (dashed red)
        if feedback_set:
            nx.draw_networkx_edges(
                G, pos, edgelist=list(feedback_set),
                width=1.8, alpha=0.8, edge_color="#e74c3c",
                style="dashed", arrows=True, arrowsize=14, arrowstyle="-|>",
                connectionstyle="arc3,rad=0.15",
                min_source_margin=18, min_target_margin=18,
                ax=ax,
            )

        # Draw nodes coloured by category
        for cat in NAS_CATEGORIES:
            cat_roles = [r for r in ROLES_ORDERED if ROLE_TO_CATEGORY[r] == cat]
            nx.draw_networkx_nodes(
                G, pos, nodelist=cat_roles,
                node_size=700, node_color=CATEGORY_PALETTE[cat],
                edgecolors="black", linewidths=1.2, ax=ax,
            )

        # Short labels
        short = {}
        for r in ROLES_ORDERED:
            parts = r.split("_", 1)
            short[r] = parts[1] if len(parts) > 1 else r
        nx.draw_networkx_labels(G, pos, short, font_size=7, font_weight="bold", ax=ax)

        # Title & metrics
        title = f"{TOPOLOGY_LABELS.get(topo, topo)}"
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

        info = (
            f"Edges: {metrics['n_edges']}  |  "
            f"Feedback: {metrics['n_feedback']}  |  "
            f"Density: {metrics['density']:.3f}"
        )
        ax.text(0.5, -0.02, info, ha="center", va="top",
                transform=ax.transAxes, fontsize=10, color="#444444")

        ax.set_xlim(-6, 6)
        ax.set_ylim(-5.5, 5.5)
        ax.axis("off")

    # Category legend
    handles = [
        mpatches.Patch(color=CATEGORY_PALETTE[c], label=c.capitalize())
        for c in NAS_CATEGORIES
    ]
    handles.append(
        plt.Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=2,
                   label="Feedback loop")
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11,
               frameon=True, fancybox=True)

    plt.subplots_adjust(bottom=0.08, wspace=0.05)
    _save(fig, out_path / f"fig1_topology_comparison.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 2: Pheromone Heatmap
# ---------------------------------------------------------------------------

def _build_pheromone_matrix(
    all_data: dict,
    topology: str,
) -> tuple[np.ndarray, list[str]]:
    """Build 18×18 mean-pheromone matrix (roles ordered)."""
    n = len(ROLES_ORDERED)
    accum = np.zeros((n, n))
    count = 0

    role_idx = {r: i for i, r in enumerate(ROLES_ORDERED)}

    for (topo, ds), entry in all_data.items():
        if topo != topology:
            continue
        for run in entry["runs"]:
            ph = run.get("pheromones_display", {})
            for key, val in ph.items():
                parts = key.split("->")
                if len(parts) == 2 and parts[0] in role_idx and parts[1] in role_idx:
                    accum[role_idx[parts[0]], role_idx[parts[1]]] += val
            count += 1

    if count > 0:
        accum /= count

    return accum, ROLES_ORDERED


def _build_category_flow(mat: np.ndarray) -> np.ndarray:
    """Compress 18×18 → 5×5 category flow matrix."""
    n_cats = len(NAS_CATEGORIES)
    cat_idx = {c: i for i, c in enumerate(NAS_CATEGORIES)}
    flow = np.zeros((n_cats, n_cats))

    role_idx = {r: i for i, r in enumerate(ROLES_ORDERED)}
    for r_src, i_src in role_idx.items():
        for r_tgt, i_tgt in role_idx.items():
            c_src = cat_idx[ROLE_TO_CATEGORY[r_src]]
            c_tgt = cat_idx[ROLE_TO_CATEGORY[r_tgt]]
            flow[c_src, c_tgt] += mat[i_src, i_tgt]
    return flow


def fig2_pheromone_heatmap(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """Pheromone heatmap: 18×18 role-level + 5×5 category-level."""
    topologies = ["hardcoded", "gpt-5.2", "gpt-4o-mini"]

    # Panel A: two 18×18 heatmaps (hardcoded, gpt-5.2)
    # Panel B: three 5×5 category flow heatmaps
    fig = plt.figure(figsize=(24, 16))

    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.35, wspace=0.30)

    # Short role labels for axis ticks
    short_labels = []
    for r in ROLES_ORDERED:
        parts = r.split("_", 1)
        short_labels.append(parts[1] if len(parts) > 1 else r)

    # Panel A: 18×18 heatmaps for hardcoded and gpt-5.2
    for col_idx, topo in enumerate(["hardcoded", "gpt-5.2"]):
        ax = fig.add_subplot(gs[0, col_idx])
        mat, _ = _build_pheromone_matrix(all_data, topo)

        # Mask zeros for cleaner look
        mask = mat == 0

        sns.heatmap(
            mat, mask=mask, ax=ax,
            xticklabels=short_labels, yticklabels=short_labels,
            cmap="YlOrRd", linewidths=0.5, linecolor="white",
            annot=np.where(mat > 0.5, np.round(mat, 2), np.nan),
            fmt=".2f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.8, "label": "Mean Pheromone"},
            vmin=0, vmax=1.0,
        )

        # Draw category separators
        boundaries = [0, 4, 8, 12, 16, 18]
        for b in boundaries[1:-1]:
            ax.axhline(b, color="black", linewidth=1.5)
            ax.axvline(b, color="black", linewidth=1.5)

        ax.set_title(f"{TOPOLOGY_LABELS[topo]} (18×18 Roles)",
                     fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7)

    # Difference map (gpt-5.2 − hardcoded)
    ax_diff = fig.add_subplot(gs[0, 2])
    mat_hc, _ = _build_pheromone_matrix(all_data, "hardcoded")
    mat_52, _ = _build_pheromone_matrix(all_data, "gpt-5.2")
    diff = mat_52 - mat_hc
    mask_diff = (mat_52 == 0) & (mat_hc == 0)

    sns.heatmap(
        diff, mask=mask_diff, ax=ax_diff,
        xticklabels=short_labels, yticklabels=short_labels,
        cmap="RdBu_r", center=0, linewidths=0.5, linecolor="white",
        annot=np.where(np.abs(diff) > 0.1, np.round(diff, 2), np.nan),
        fmt="+.2f", annot_kws={"size": 6},
        cbar_kws={"shrink": 0.8, "label": "Δ Pheromone"},
    )
    for b in boundaries[1:-1]:
        ax_diff.axhline(b, color="black", linewidth=1.5)
        ax_diff.axvline(b, color="black", linewidth=1.5)
    ax_diff.set_title("Δ (GPT-5.2 − Hardcoded)", fontsize=14, fontweight="bold")
    ax_diff.tick_params(axis="both", labelsize=7)

    # Panel B: 5×5 category flow
    cat_labels = [c.capitalize() for c in NAS_CATEGORIES]
    for col_idx, topo in enumerate(topologies):
        ax_cat = fig.add_subplot(gs[1, col_idx])
        mat_full, _ = _build_pheromone_matrix(all_data, topo)
        flow = _build_category_flow(mat_full)

        sns.heatmap(
            flow, ax=ax_cat,
            xticklabels=cat_labels, yticklabels=cat_labels,
            cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 10},
            linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Flow"},
        )
        ax_cat.set_title(f"{TOPOLOGY_LABELS[topo]} (Category Flow)",
                         fontsize=12, fontweight="bold")
        ax_cat.tick_params(axis="both", labelsize=9)

    fig.suptitle("Pheromone Analysis: Learned Knowledge in the KG",
                 fontsize=18, fontweight="bold", y=1.01)

    _save(fig, out_path / f"fig2_pheromone_heatmap.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 3: Entropy Comparison
# ---------------------------------------------------------------------------

def fig3_entropy_comparison(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """Violin plot of pheromone entropy by topology."""
    records: list[dict] = []

    for (topo, ds), entry in all_data.items():
        ph_stats = compute_pheromone_stats(entry["runs"])
        for ent in ph_stats["entropies"]:
            records.append({"Topology": TOPOLOGY_LABELS.get(topo, topo),
                            "Entropy": ent, "topology_key": topo})

    if not records:
        print("  [skip] fig3: no data")
        return

    import pandas as pd
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 6))

    order = [TOPOLOGY_LABELS[t] for t in ["hardcoded", "gpt-5.2", "gpt-4o-mini"]
             if TOPOLOGY_LABELS[t] in df["Topology"].unique()]
    palette_dict = {TOPOLOGY_LABELS[t]: TOPOLOGY_PALETTE[t]
                    for t in ["hardcoded", "gpt-5.2", "gpt-4o-mini"]
                    if TOPOLOGY_LABELS[t] in df["Topology"].unique()}

    sns.violinplot(
        data=df, x="Topology", y="Entropy", hue="Topology", order=order,
        palette=palette_dict, inner="box", linewidth=1.2, legend=False, ax=ax,
    )
    sns.stripplot(
        data=df, x="Topology", y="Entropy", order=order,
        color="black", alpha=0.15, size=2, jitter=True, ax=ax,
    )

    # Wilcoxon p-values between pairs
    topo_keys_present = [t for t in ["hardcoded", "gpt-5.2", "gpt-4o-mini"]
                         if TOPOLOGY_LABELS[t] in df["Topology"].unique()]
    pairs = []
    for i in range(len(topo_keys_present)):
        for j in range(i + 1, len(topo_keys_present)):
            pairs.append((topo_keys_present[i], topo_keys_present[j]))

    y_max = df["Entropy"].max()
    for k, (t1, t2) in enumerate(pairs):
        v1 = df[df["Topology"] == TOPOLOGY_LABELS[t1]]["Entropy"].values
        v2 = df[df["Topology"] == TOPOLOGY_LABELS[t2]]["Entropy"].values
        if len(v1) > 1 and len(v2) > 1:
            try:
                stat, p = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
            except ValueError:
                p = 1.0
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            i1 = topo_keys_present.index(t1)
            i2 = topo_keys_present.index(t2)
            y_bar = y_max + 0.15 + k * 0.25
            ax.plot([i1, i1, i2, i2], [y_bar - 0.05, y_bar, y_bar, y_bar - 0.05],
                    color="black", linewidth=1)
            ax.text((i1 + i2) / 2, y_bar + 0.02, f"{stars} (p={p:.1e})",
                    ha="center", fontsize=9)

    ax.set_ylabel("Shannon Entropy (bits)", fontsize=13)
    ax.set_xlabel("")
    ax.set_title("Pheromone Entropy: ACO Selectivity by Topology",
                 fontsize=15, fontweight="bold")

    _save(fig, out_path / f"fig3_entropy_comparison.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 4: Role Frequency & Category Allocation
# ---------------------------------------------------------------------------

def fig4_role_frequency(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """Bar chart of role frequency + radar chart of category allocation."""
    import pandas as pd

    topologies = ["hardcoded", "gpt-5.2", "gpt-4o-mini"]

    # Aggregate mean role frequency per topology
    topo_role_freq: dict[str, dict[str, float]] = {}
    for topo in topologies:
        accum = {r: [] for r in ROLES_ORDERED}
        for (t, ds), entry in all_data.items():
            if t != topo:
                continue
            rs = compute_role_stats(entry["runs"])
            for r in ROLES_ORDERED:
                accum[r].append(rs["mean_freq"][r])
        topo_role_freq[topo] = {r: np.mean(v) if v else 0 for r, v in accum.items()}

    # ---- Panel A: grouped bar chart ----
    fig, (ax_bar, ax_radar) = plt.subplots(1, 2, figsize=(20, 8),
                                            gridspec_kw={"width_ratios": [2, 1]})

    # Build bar data
    short_labels = [r.split("_", 1)[1] for r in ROLES_ORDERED]
    x = np.arange(len(ROLES_ORDERED))
    width = 0.25

    present_topos = [t for t in topologies if t in topo_role_freq and
                     any(v > 0 for v in topo_role_freq[t].values())]

    for i, topo in enumerate(present_topos):
        vals = [topo_role_freq[topo][r] for r in ROLES_ORDERED]
        bars = ax_bar.bar(x + i * width, vals, width,
                          label=TOPOLOGY_LABELS[topo],
                          color=TOPOLOGY_PALETTE[topo], alpha=0.85,
                          edgecolor="white", linewidth=0.5)

    # Category background shading
    boundaries = [0, 4, 8, 12, 16, 18]
    for k, cat in enumerate(NAS_CATEGORIES):
        ax_bar.axvspan(boundaries[k] - 0.4, boundaries[k + 1] - 0.6,
                       alpha=0.08, color=CATEGORY_PALETTE[cat])

    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("Mean Frequency (visits per run)", fontsize=12)
    ax_bar.set_title("Role Frequency by Topology", fontsize=14, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)

    # ---- Panel B: radar chart of category allocation ----
    cat_freq: dict[str, dict[str, float]] = {}
    for topo in present_topos:
        cat_freq[topo] = {}
        for cat in NAS_CATEGORIES:
            roles_in_cat = [r for r in ROLES_ORDERED if ROLE_TO_CATEGORY[r] == cat]
            cat_freq[topo][cat] = sum(topo_role_freq[topo][r] for r in roles_in_cat)

    # Normalise to percentage
    for topo in present_topos:
        total = sum(cat_freq[topo].values())
        if total > 0:
            for cat in NAS_CATEGORIES:
                cat_freq[topo][cat] = 100 * cat_freq[topo][cat] / total

    # Radar
    ax_radar.remove()
    ax_radar = fig.add_subplot(122, polar=True)

    categories_labels = [c.capitalize() for c in NAS_CATEGORIES]
    n_cats = len(NAS_CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for topo in present_topos:
        values = [cat_freq[topo][c] for c in NAS_CATEGORIES]
        values += values[:1]
        ax_radar.plot(angles, values, linewidth=2, label=TOPOLOGY_LABELS[topo],
                      color=TOPOLOGY_PALETTE[topo])
        ax_radar.fill(angles, values, alpha=0.15, color=TOPOLOGY_PALETTE[topo])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories_labels, fontsize=10)
    ax_radar.set_title("Category Allocation (%)", fontsize=14, fontweight="bold",
                       pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.subplots_adjust(wspace=0.35)
    _save(fig, out_path / f"fig4_role_frequency.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 5: Dominant Path Analysis
# ---------------------------------------------------------------------------

def fig5_path_analysis(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """Timeline-style diagram of top-10 dominant paths for gpt-5.2."""
    # Prefer gpt-5.2, fallback to any available topology
    target_topo = "gpt-5.2"
    available_topos = {t for (t, _) in all_data}
    if target_topo not in available_topos:
        target_topo = next(iter(available_topos), None)
    if target_topo is None:
        print("  [skip] fig5: no data")
        return

    # Aggregate paths across all datasets for target topology
    path_counter: Counter = Counter()
    for (topo, ds), entry in all_data.items():
        if topo != target_topo:
            continue
        for run in entry["runs"]:
            for p in run.get("top_paths", []):
                path_tuple = tuple(p["path"])
                path_counter[path_tuple] += p.get("count", 1)

    top10 = path_counter.most_common(10)
    if not top10:
        print("  [skip] fig5: no paths")
        return

    max_count = top10[0][1]
    max_len = max(len(p) for p, _ in top10)

    fig, ax = plt.subplots(figsize=(max(16, max_len * 1.8), len(top10) * 0.9 + 1))

    box_w = 1.4
    box_h = 0.5
    y_gap = 0.85
    x_gap = 0.15

    for row, (path, count) in enumerate(top10):
        y = (len(top10) - 1 - row) * y_gap

        for col, role in enumerate(path):
            x = col * (box_w + x_gap)
            cat = ROLE_TO_CATEGORY.get(role, "topology")
            color = CATEGORY_PALETTE[cat]

            short = role.split("_", 1)[1] if "_" in role else role

            fancy = mpatches.FancyBboxPatch(
                (x, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
            )
            ax.add_patch(fancy)
            ax.text(x + box_w / 2, y, short,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="black")

            # Arrow between boxes
            if col < len(path) - 1:
                ax.annotate(
                    "", xy=(x + box_w + x_gap * 0.15, y),
                    xytext=(x + box_w - 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2),
                )

        # Frequency bar on the right
        bar_x = (max_len) * (box_w + x_gap) + 0.5
        bar_w = 3.0 * count / max_count
        ax.barh(y, bar_w, height=box_h * 0.6, left=bar_x,
                color=TOPOLOGY_PALETTE.get(target_topo, "#888888"), alpha=0.7,
                edgecolor="white")
        ax.text(bar_x + bar_w + 0.15, y, str(count),
                ha="left", va="center", fontsize=9, fontweight="bold")

    # Rank labels
    for row in range(len(top10)):
        y = (len(top10) - 1 - row) * y_gap
        ax.text(-0.5, y, f"#{row + 1}", ha="right", va="center", fontsize=10,
                fontweight="bold", color="#444444")

    ax.set_xlim(-0.8, (max_len) * (box_w + x_gap) + 5)
    ax.set_ylim(-0.6, (len(top10) - 0.5) * y_gap + 0.5)
    ax.axis("off")

    # Category legend
    handles = [mpatches.Patch(color=CATEGORY_PALETTE[c], label=c.capitalize())
               for c in NAS_CATEGORIES]
    ax.legend(handles=handles, loc="lower right", fontsize=10, ncol=5,
              frameon=True, fancybox=True)

    ax.set_title(
        f"Top-10 Dominant Paths ({TOPOLOGY_LABELS.get(target_topo, target_topo)}, all datasets)",
        fontsize=15, fontweight="bold", pad=15,
    )

    _save(fig, out_path / f"fig5_path_analysis.{fmt}", fmt)


# ---------------------------------------------------------------------------
# Figure 6: Cross-Dataset Stability
# ---------------------------------------------------------------------------

def fig6_cross_dataset_stability(
    all_data: dict,
    out_path: Path,
    fmt: str,
) -> None:
    """Cross-dataset pheromone correlation heatmap."""
    import pandas as pd

    topologies = ["hardcoded", "gpt-5.2", "gpt-4o-mini"]

    fig, axes_arr = plt.subplots(1, 3, figsize=(22, 7))

    corr_means: dict[str, float] = {}
    corr_stds: dict[str, float] = {}

    for idx, topo in enumerate(topologies):
        ax = axes_arr[idx]

        # Gather mean pheromone vector per dataset
        datasets_avail = sorted(
            [ds for (t, ds) in all_data if t == topo],
            key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99,
        )

        if len(datasets_avail) < 2:
            ax.text(0.5, 0.5, f"< 2 datasets\nfor {topo}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(TOPOLOGY_LABELS.get(topo, topo), fontsize=14, fontweight="bold")
            continue

        # Build vectors: common edge set across datasets
        all_edge_keys: set[str] = set()
        for ds in datasets_avail:
            entry = all_data[(topo, ds)]
            for run in entry["runs"]:
                all_edge_keys.update(run.get("pheromones_display", {}).keys())
        edge_list = sorted(all_edge_keys)

        ds_vectors: dict[str, np.ndarray] = {}
        for ds in datasets_avail:
            entry = all_data[(topo, ds)]
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

        # Correlation matrix
        n_ds = len(datasets_avail)
        corr_mat = np.ones((n_ds, n_ds))
        for i in range(n_ds):
            for j in range(i + 1, n_ds):
                r, _ = sp_stats.pearsonr(
                    ds_vectors[datasets_avail[i]],
                    ds_vectors[datasets_avail[j]],
                )
                corr_mat[i, j] = r
                corr_mat[j, i] = r

        # Extract off-diagonal correlations for summary stats
        offdiag = corr_mat[np.triu_indices(n_ds, k=1)]
        corr_means[topo] = np.mean(offdiag)
        corr_stds[topo] = np.std(offdiag)

        # Plot heatmap
        sns.heatmap(
            corr_mat, ax=ax,
            xticklabels=datasets_avail, yticklabels=datasets_avail,
            cmap="coolwarm", vmin=0.5, vmax=1.0,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8},
        )

        # Draw group separators
        # Find boundaries between dataset groups
        group_boundaries = []
        prev_group = None
        for i, ds in enumerate(datasets_avail):
            for gname, gds in CATEGORY_GROUPS.items():
                if ds in gds:
                    cur_group = gname
                    break
            else:
                cur_group = "other"
            if prev_group is not None and cur_group != prev_group:
                group_boundaries.append(i)
            prev_group = cur_group

        for b in group_boundaries:
            ax.axhline(b, color="black", linewidth=2)
            ax.axvline(b, color="black", linewidth=2)

        mean_r = np.mean(offdiag)
        ax.set_title(
            f"{TOPOLOGY_LABELS.get(topo, topo)}\n(mean r = {mean_r:.3f})",
            fontsize=13, fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        "Cross-Dataset Pheromone Correlation (Transferability Evidence)",
        fontsize=16, fontweight="bold", y=1.02,
    )

    plt.subplots_adjust(wspace=0.35)
    _save(fig, out_path / f"fig6_cross_dataset_stability.{fmt}", fmt)

    # Print summary
    print("\n  Cross-dataset correlation summary:")
    for topo in topologies:
        if topo in corr_means:
            print(f"    {TOPOLOGY_LABELS[topo]:>12s}: "
                  f"mean r = {corr_means[topo]:.3f} ± {corr_stds[topo]:.3f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural analysis of NAS Knowledge Graph — figures for paper"
    )
    parser.add_argument(
        "--results-dir",
        default="results/nas_bench_graph",
        help="Directory with nasbench_graph_llm_sweep_*.json files",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/nas_bench_graph/figures",
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    _setup_style()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    all_data = load_all_results(args.results_dir)
    print(f"  Loaded {len(all_data)} (topology, dataset) entries")

    # Summary
    topos = sorted({t for t, _ in all_data})
    datasets = sorted({d for _, d in all_data})
    print(f"  Topologies: {topos}")
    print(f"  Datasets:   {datasets}")
    for topo in topos:
        entries = [(t, d) for (t, d) in all_data if t == topo]
        sample_entry = all_data[entries[0]]
        n_edges = len(sample_entry["runs"][0].get("pheromones_display", {}))
        n_runs = sum(len(all_data[k]["runs"]) for k in entries)
        print(f"    {topo:>12s}: {len(entries)} datasets, "
              f"{n_edges} edges, {n_runs} total runs")

    fmt = args.format

    print("\nFig 1: Topology Comparison ...")
    fig1_topology_comparison(all_data, figures_dir, fmt)

    print("Fig 2: Pheromone Heatmap ...")
    fig2_pheromone_heatmap(all_data, figures_dir, fmt)

    print("Fig 3: Entropy Comparison ...")
    fig3_entropy_comparison(all_data, figures_dir, fmt)

    print("Fig 4: Role Frequency ...")
    fig4_role_frequency(all_data, figures_dir, fmt)

    print("Fig 5: Path Analysis ...")
    fig5_path_analysis(all_data, figures_dir, fmt)

    print("Fig 6: Cross-Dataset Stability ...")
    fig6_cross_dataset_stability(all_data, figures_dir, fmt)

    print(f"\nAll figures saved to: {figures_dir}/")


if __name__ == "__main__":
    main()
