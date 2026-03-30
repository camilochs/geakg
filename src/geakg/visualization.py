"""Visualization module for NS-SE.

Provides domain-agnostic visualizations for:
- AKG operator graphs (pheromone levels, synthesized vs generic)
- Convergence curves with synthesized operator appearances
- MetaGraph topology

Usage:
    from src.geakg.visualization import visualize_akg, plot_convergence

    # Visualize AKG with operators
    visualize_akg(
        operator_pheromones=selector.operator_pheromones,
        output_path="akg_visualization.png",
        title="AKG After Optimization",
        meta_graph=meta_graph,
    )

    # Plot convergence with synthesized markers
    plot_convergence(
        result=experiment_result,
        output_path="convergence.png",
    )
"""

import re
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from loguru import logger

from src.geakg.execution import is_synth_operator


def visualize_metagraph_topology(
    meta_graph: Any,
    output_path: str | Path,
    title: str = "MetaGraph Topology",
) -> None:
    """Visualize the metagraph topology designed by LLM.

    Shows:
    - All roles as nodes (color-coded by category)
    - All edges with weights and conditions
    - Reasoning from LLM (if available)

    Args:
        meta_graph: MetaGraph object with nodes and edges
        output_path: Path to save the visualization
        title: Title for the plot
    """
    if meta_graph is None:
        logger.warning("No metagraph to visualize")
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for role in meta_graph.nodes.keys():
        role_str = role.value if hasattr(role, "value") else str(role)
        G.add_node(role_str)

    # Add edges with attributes
    edge_labels = {}
    edge_conditions = {}
    for (src_role, tgt_role), edge in meta_graph.edges.items():
        src = src_role.value if hasattr(src_role, "value") else str(src_role)
        tgt = tgt_role.value if hasattr(tgt_role, "value") else str(tgt_role)
        G.add_edge(src, tgt, weight=edge.weight)
        edge_labels[(src, tgt)] = f"{edge.weight:.2f}"

        # Track conditions
        if edge.conditions:
            cond_strs = []
            for c in edge.conditions:
                cond_type = c.condition_type.value if hasattr(c.condition_type, "value") else str(c.condition_type)
                cond_strs.append(f"{cond_type}≥{c.threshold}")
            edge_conditions[(src, tgt)] = ", ".join(cond_strs)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Layout - organize by role category
    pos = {}
    nodes = list(G.nodes())
    const_roles = sorted([r for r in nodes if r.startswith("const")])
    ls_roles = sorted([r for r in nodes if r.startswith("ls")])
    pert_roles = sorted([r for r in nodes if r.startswith("pert")])
    meta_roles = sorted([r for r in nodes if r.startswith("meta")])

    y_positions = {"const": 3, "ls": 1, "pert": -1, "meta": -3}

    def place_roles(role_list: list, y: float, x_start: float = 0) -> None:
        n = len(role_list)
        x_spacing = 2.5
        x_offset = -(n - 1) * x_spacing / 2 + x_start
        for i, role in enumerate(role_list):
            pos[role] = (x_offset + i * x_spacing, y)

    place_roles(const_roles, y_positions["const"])
    place_roles(ls_roles, y_positions["ls"])
    place_roles(pert_roles, y_positions["pert"], x_start=-0.5)
    place_roles(meta_roles, y_positions["meta"], x_start=0.5)

    # Draw edges - color by type
    for (src, tgt) in G.edges():
        # Determine edge color by transition type
        if src.startswith("const") and tgt.startswith("ls"):
            color = "#0CF574"  # Dark Cyan - initialization
        elif src.startswith("ls") and tgt.startswith("ls"):
            color = "#2F97C1"  # Dusty Denim - intensification
        elif src.startswith("ls") and tgt.startswith("pert"):
            color = "#1CCAD8"  # Slate Blue - escape
        elif src.startswith("pert") and tgt.startswith("ls"):
            color = "#15E6CD"  # Pacific Blue - re-optimization
        else:
            color = "#95A5A6"  # Gray - other

        weight = G[src][tgt]["weight"]
        width = 1 + weight * 3

        # Curve for bidirectional edges
        if G.has_edge(tgt, src):
            rad = 0.2 if src < tgt else -0.2
        else:
            rad = 0.1

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(src, tgt)],
            width=width,
            alpha=0.8,
            edge_color=color,
            arrows=True,
            arrowsize=20,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            min_source_margin=25,
            min_target_margin=25,
            ax=ax,
        )

        # Add condition label if exists
        if (src, tgt) in edge_conditions:
            src_pos, tgt_pos = pos[src], pos[tgt]
            mid_x = (src_pos[0] + tgt_pos[0]) / 2 + rad * 0.5
            mid_y = (src_pos[1] + tgt_pos[1]) / 2 + 0.3
            ax.annotate(
                edge_conditions[(src, tgt)],
                xy=(mid_x, mid_y),
                fontsize=6,
                color=color,
                ha="center",
                style="italic",
            )

    # Draw nodes
    node_colors = []
    for role in G.nodes():
        if role.startswith("const"):
            node_colors.append("#0CF574")  # Dark Cyan
        elif role.startswith("ls"):
            node_colors.append("#2F97C1")  # Dusty Denim
        elif role.startswith("pert"):
            node_colors.append("#1CCAD8")  # Slate Blue
        else:
            node_colors.append("#15E6CD")  # Pacific Blue

    nx.draw_networkx_nodes(
        G, pos,
        node_size=3000,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    # Node labels
    labels = {role: role.replace("_", "\n") for role in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    # Edge weight labels
    for (src, tgt), label in edge_labels.items():
        src_pos, tgt_pos = pos[src], pos[tgt]
        # Position label near the middle of edge
        mid_x = src_pos[0] * 0.6 + tgt_pos[0] * 0.4
        mid_y = src_pos[1] * 0.6 + tgt_pos[1] * 0.4
        ax.annotate(
            label,
            xy=(mid_x, mid_y),
            fontsize=7,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
        )

    # Legend for edge types
    legend_elements = [
        mpatches.Patch(color="#0CF574", label="Construction"),
        mpatches.Patch(color="#2F97C1", label="Local Search"),
        mpatches.Patch(color="#1CCAD8", label="Perturbation"),
        plt.Line2D([0], [0], color="#0CF574", linewidth=2, label="const → ls"),
        plt.Line2D([0], [0], color="#2F97C1", linewidth=2, label="ls → ls"),
        plt.Line2D([0], [0], color="#1CCAD8", linewidth=2, label="ls → pert (escape)"),
        plt.Line2D([0], [0], color="#15E6CD", linewidth=2, label="pert → ls (re-opt)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Add reasoning text if available
    reasoning = getattr(meta_graph, "llm_reasoning", None) or getattr(meta_graph, "description", None)
    if reasoning:
        # Truncate if too long
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        ax.text(
            0.02, 0.02, f"Design: {reasoning}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            wrap=True,
        )

    # Title with stats
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    full_title = f"{title}\n{n_nodes} roles, {n_edges} edges"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=10)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"MetaGraph topology saved to: {output_path}")


def visualize_akg(
    operator_pheromones: dict[tuple[str, str], float],
    output_path: str | Path,
    title: str = "AKG Operator Graph",
    meta_graph: Any = None,
) -> None:
    """Visualize the AKG with generic and synthesized-synthesized operators.

    Creates a visualization showing:
    - Roles as nodes (color-coded by category)
    - Edges between roles with weights from MetaGraph
    - Operator annotations showing pheromone levels
    - synthesized vs generic operator distinction

    Args:
        operator_pheromones: Dict mapping (role, operator) to pheromone level.
        output_path: Path to save the visualization image.
        title: Title for the visualization.
        meta_graph: Optional MetaGraph for edge information.
    """
    # Group operators by role
    roles: dict[str, list[tuple[str, float]]] = {}
    for (role, op), tau in operator_pheromones.items():
        if role not in roles:
            roles[role] = []
        roles[role].append((op, tau))

    if not roles:
        logger.warning("No operators to visualize")
        return

    # Create directed graph
    G = nx.DiGraph()
    for role in roles:
        G.add_node(role)

    # Add edges from meta_graph
    edge_weights = {}
    if meta_graph is not None:
        for (src_role, tgt_role), edge in meta_graph.edges.items():
            src = src_role.value if hasattr(src_role, "value") else str(src_role)
            tgt = tgt_role.value if hasattr(tgt_role, "value") else str(tgt_role)
            if src in roles and tgt in roles:
                G.add_edge(src, tgt)
                edge_weights[(src, tgt)] = edge.weight

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    # Layout - organize by role category
    pos = {}
    role_list = list(roles.keys())
    const_roles = [r for r in role_list if r.startswith("const")]
    ls_roles = [r for r in role_list if r.startswith("ls")]
    pert_roles = [r for r in role_list if r.startswith("pert")]
    meta_roles = [r for r in role_list if r.startswith("meta")]

    y_positions = {"const": 3, "ls": 1, "pert": -1, "meta": -3}

    def place_roles(role_list: list, y: float, x_start: float = 0) -> None:
        n = len(role_list)
        x_spacing = 3.0
        x_offset = -(n - 1) * x_spacing / 2 + x_start
        for i, role in enumerate(sorted(role_list)):
            pos[role] = (x_offset + i * x_spacing, y)

    place_roles(const_roles, y_positions["const"])
    place_roles(ls_roles, y_positions["ls"])
    place_roles(pert_roles, y_positions["pert"], x_start=-1)
    place_roles(meta_roles, y_positions["meta"], x_start=1)

    # Draw edges with visible arrows
    for (src, tgt), weight in edge_weights.items():
        if src in pos and tgt in pos:
            # Determine curve direction for bidirectional edges
            reverse_key = (tgt, src)
            if reverse_key in edge_weights:
                rad = 0.2 if src < tgt else -0.2
            else:
                rad = 0.15

            width = 0.5 + weight * 3
            alpha = 0.3 + weight * 0.5
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(src, tgt)],
                width=width,
                alpha=min(alpha, 0.9),
                edge_color="gray",
                arrows=True,
                arrowsize=20,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                min_source_margin=30,
                min_target_margin=30,
                ax=ax,
            )

    # Draw nodes with category-based colors
    node_colors = []
    for role in G.nodes():
        if role.startswith("const"):
            node_colors.append("#0CF574")  # Dark Cyan
        elif role.startswith("ls"):
            node_colors.append("#2F97C1")  # Dusty Denim
        elif role.startswith("pert"):
            node_colors.append("#1CCAD8")  # Slate Blue
        else:
            node_colors.append("#15E6CD")  # Pacific Blue

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=4000,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    labels = {role: role.replace("_", "\n") for role in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    # Operator info annotations
    for role, ops in roles.items():
        if role not in pos:
            continue
        rx, ry = pos[role]
        ops_sorted = sorted(ops, key=lambda x: x[1], reverse=True)

        op_lines = []
        n_synth = sum(1 for op, _ in ops if is_synth_operator(op))
        n_generic = len(ops) - n_synth

        if n_synth > 0:
            op_lines.append(f"--- {n_synth} synthesized synthesized ---")
        op_lines.append(f"--- {n_generic} generic ---")

        for op, tau in ops_sorted[:5]:
            prefix = "[synthesized] " if is_synth_operator(op) else "[G] "
            op_short = op[:16] + ".." if len(op) > 16 else op
            op_lines.append(f"{prefix}{op_short}: {tau:.3f}")

        if len(ops_sorted) > 5:
            op_lines.append(f"  +{len(ops_sorted) - 5} more...")

        op_text = "\n".join(op_lines)
        ax.annotate(
            op_text,
            xy=(rx + 0.8, ry),
            fontsize=6,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray",
            ),
            ha="left",
            va="center",
        )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#0CF574", label="Construction"),
        mpatches.Patch(color="#2F97C1", label="Local Search"),
        mpatches.Patch(color="#1CCAD8", label="Perturbation"),
        mpatches.Patch(color="#15E6CD", label="Meta-heuristic"),
        mpatches.Patch(color="white", edgecolor="#1CCAD8", linewidth=2, label="[synthesized] Synthesized"),
        mpatches.Patch(color="white", edgecolor="#2F97C1", linewidth=2, label="[G] Generic"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_xlim(-8, 10)
    ax.set_ylim(-5, 5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"AKG visualization saved to: {output_path}")


def plot_convergence(
    result: dict,
    output_path: str | Path,
    title_prefix: str = "NS-SE Convergence",
    multi_instance: bool = False,
    n_instances: int = 0,
) -> None:
    """Plot convergence curve with synthesized operator appearances highlighted.

    Creates a publication-quality figure showing:
    - Convergence curve (gap to optimal over time)
    - Vertical lines marking when synthesized operators first appear in best path
    - Shaded region showing "synthesized-assisted" phase

    Args:
        result: Experiment result dict containing:
            - convergence_history: list of (time, gap) tuples for multi-instance
              or list of (time, fitness) tuples for single-instance
            - optimal: known optimal value (only for single-instance)
            - synth_first_appearance: list of dicts or dict of {op_id: (time, fitness, role)}
        output_path: Path to save the plot.
        title_prefix: Prefix for the plot title.
        multi_instance: If True, convergence_history contains gaps directly.
        n_instances: Number of instances (for multi-instance mode label).
    """
    history = result.get("convergence_history", [])
    if not history:
        logger.warning("No convergence history to plot")
        return

    optimal = result.get("optimal", 1)
    synth_appearances_raw = result.get("synth_first_appearance", {})

    # Normalize synth_appearances format
    # Can be list of dicts or dict of tuples
    synth_appearances = {}
    if isinstance(synth_appearances_raw, list):
        # Multi-instance format: list of {"operator_id", "time", "fitness", "role"}
        for item in synth_appearances_raw:
            op_id = item.get("operator_id", "unknown")
            t = item.get("time", 0)
            fitness = item.get("fitness", 0)  # This is already a gap in multi-instance
            role = item.get("role", "unknown")
            synth_appearances[op_id] = (t, fitness, role)
    else:
        # Single-instance format: dict of {op_id: (time, fitness, role)}
        synth_appearances = synth_appearances_raw

    # Handle convergence_history format (can be list of dicts or list of tuples)
    if history and isinstance(history[0], dict):
        times = [h.get("time", 0) for h in history]
        if multi_instance:
            gaps = [h.get("value", 0) for h in history]
        else:
            gaps = [100 * (h.get("value", 0) - optimal) / optimal for h in history]
    else:
        times = [t for t, _ in history]
        if multi_instance:
            gaps = [g for _, g in history]
        else:
            gaps = [100 * (f - optimal) / optimal for _, f in history]

    # Set up figure with style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Find first synthesized appearance time for shading
    first_synth_time = None
    if synth_appearances:
        first_synth_time = min(t for t, _, _ in synth_appearances.values())

    # Shade the synthesized-assisted region
    if first_synth_time is not None and times:
        ax.axvspan(
            first_synth_time,
            times[-1],
            alpha=0.15,
            color="purple",
            label="synthesized-Assisted Phase",
            zorder=1,
        )

    # Plot convergence curve
    ax.plot(
        times,
        gaps,
        color="#2E86AB",
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Best Solution Gap",
        zorder=3,
    )

    # Add synthesized appearance markers
    synth_colors = plt.cm.Set2(range(len(synth_appearances)))
    for idx, (op_id, (t, fitness_or_gap, role)) in enumerate(synth_appearances.items()):
        # In multi-instance mode, fitness_or_gap is already a gap percentage
        # In single-instance mode, convert fitness to gap
        if multi_instance:
            gap_at_time = fitness_or_gap
        else:
            gap_at_time = 100 * (fitness_or_gap - optimal) / optimal

        # Vertical line
        ax.axvline(
            x=t,
            color=synth_colors[idx],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )

        # Marker on the curve
        ax.scatter(
            [t],
            [gap_at_time],
            s=150,
            color=synth_colors[idx],
            marker="*",
            edgecolors="black",
            linewidths=1,
            zorder=4,
        )

        # Extract heuristic name from operator ID
        heuristic_name = _extract_heuristic_name(op_id)

        # Format role name nicely
        role_display = role.replace("_", " ").title()

        # Annotation with operator info - position relative to curve, not absolute
        y_offset = 15 + (idx % 3) * 25  # Stagger annotations vertically
        ax.annotate(
            f"★ {heuristic_name}\n   [{role_display}]",
            xy=(t, gap_at_time),
            xytext=(8, y_offset),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=synth_colors[idx],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=synth_colors[idx],
                alpha=0.9,
            ),
            arrowprops=dict(arrowstyle="->", color=synth_colors[idx], lw=1.2),
            zorder=5,
        )

    # Final gap annotation
    if gaps:
        ax.annotate(
            f"Final: {gaps[-1]:.2f}%",
            xy=(times[-1], gaps[-1]),
            xytext=(-60, -30),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#E8F4F8",
                edgecolor="#2E86AB",
            ),
            arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.5),
        )

    # Labels and title
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    if multi_instance:
        ylabel = f"Average Gap to Optimal (%) over {n_instances} instances"
    else:
        ylabel = "Gap to Optimal (%)"
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    title = f"{title_prefix}: Generic Operators → synthesized Synthesis"
    if multi_instance and n_instances > 0:
        title += f"\n(Training on {n_instances} instances with Instance Hardness Sampling)"
    if synth_appearances:
        title += f"\n({len(synth_appearances)} synthesized operators contributed to best path)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend
    gap_label = "Best Avg Gap" if multi_instance else "Best Solution Gap"
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="#2E86AB",
            linewidth=2.5,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            label=gap_label,
        ),
    ]
    if first_synth_time is not None:
        legend_elements.append(
            mpatches.Patch(color="purple", alpha=0.15, label="synthesized-Assisted Phase")
        )
    if synth_appearances:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="gray",
                linestyle="None",
                markersize=12,
                label="synthesized First in Best Path",
            )
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.95)

    # Grid styling
    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis to start at 0 if gaps are small
    if gaps and max(gaps) < 20:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    logger.info(f"Convergence plot saved to: {output_path}")


def _extract_heuristic_name(op_id: str) -> str:
    """Extract heuristic name from operator ID.

    Examples:
        "two_opt_tsp_abc123" -> "two_opt"
        "or_opt_improved_tsp_def456" -> "or_opt_improved"
        "swap_jssp_xyz789" -> "swap"

    Args:
        op_id: Full operator ID with domain suffix.

    Returns:
        Extracted heuristic name.
    """
    # Try common domain patterns
    for domain in ["_tsp_", "_jssp_", "_vrp_", "_bpp_"]:
        if domain in op_id:
            return op_id.split(domain)[0]

    # Fallback: remove last hash suffix
    if re.search(r"_[a-f0-9]{6}$", op_id):
        return op_id.rsplit("_", 1)[0]

    return op_id


def plot_pheromone_evolution(
    pheromone_history: list[dict[tuple[str, str], float]],
    output_path: str | Path,
    top_k: int = 10,
    title: str = "Pheromone Evolution",
) -> None:
    """Plot pheromone evolution over iterations.

    Shows how operator pheromone levels change during optimization.

    Args:
        pheromone_history: List of pheromone dicts per iteration.
        output_path: Path to save the plot.
        top_k: Number of top operators to show.
        title: Plot title.
    """
    if not pheromone_history:
        logger.warning("No pheromone history to plot")
        return

    # Find top-k operators by final pheromone
    final_pheromones = pheromone_history[-1]
    top_ops = sorted(final_pheromones.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_keys = [k for k, _ in top_ops]

    # Extract history for top operators
    iterations = list(range(len(pheromone_history)))
    traces = {k: [] for k in top_keys}

    for pheromones in pheromone_history:
        for k in top_keys:
            traces[k].append(pheromones.get(k, 0.0))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(range(len(top_keys)))
    for (role, op), color in zip(top_keys, colors):
        label = f"{role}: {op[:15]}..."
        is_synth = is_synth_operator(op)
        linestyle = "--" if is_synth else "-"
        ax.plot(
            iterations,
            traces[(role, op)],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2 if is_synth else 1.5,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Pheromone Level", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pheromone evolution plot saved to: {output_path}")


def plot_operator_usage(
    usage_counts: dict[str, int],
    output_path: str | Path,
    title: str = "Operator Usage Distribution",
) -> None:
    """Plot operator usage distribution as horizontal bar chart.

    Args:
        usage_counts: Dict mapping operator_id to usage count.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not usage_counts:
        logger.warning("No usage data to plot")
        return

    # Sort by count
    sorted_ops = sorted(usage_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit to top 20
    sorted_ops = sorted_ops[:20]

    ops = [op for op, _ in sorted_ops]
    counts = [count for _, count in sorted_ops]

    # Color by synthesized vs generic
    colors = ["#FF6B6B" if is_synth_operator(op) else "#4ECDC4" for op in ops]

    fig, ax = plt.subplots(figsize=(10, max(6, len(ops) * 0.3)))

    y_pos = range(len(ops))
    ax.barh(y_pos, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([op[:25] + "..." if len(op) > 25 else op for op in ops], fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Usage Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#FF6B6B", label="synthesized Synthesized"),
        mpatches.Patch(color="#4ECDC4", label="Generic"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Operator usage plot saved to: {output_path}")


def plot_instance_gaps(
    per_instance_gaps: dict[str, float],
    output_path: str | Path,
    title: str = "Gap per Instance",
) -> None:
    """Plot gap for each instance as bar chart.

    Args:
        per_instance_gaps: Dict mapping instance_id to gap percentage.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not per_instance_gaps:
        logger.warning("No instance gaps to plot")
        return

    # Sort by gap (worst first)
    sorted_gaps = sorted(per_instance_gaps.items(), key=lambda x: x[1], reverse=True)

    instances = [inst for inst, _ in sorted_gaps]
    gaps = [gap for _, gap in sorted_gaps]

    # Color gradient: red (high gap) to green (low gap)
    max_gap = max(gaps) if gaps else 1
    colors = [plt.cm.RdYlGn(1 - gap / max_gap) for gap in gaps]

    fig, ax = plt.subplots(figsize=(10, max(5, len(instances) * 0.4)))

    y_pos = range(len(instances))
    bars = ax.barh(y_pos, gaps, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, gap in zip(bars, gaps):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{gap:.2f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(instances, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Gap to Optimal (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add average line
    avg_gap = sum(gaps) / len(gaps)
    ax.axvline(x=avg_gap, color="blue", linestyle="--", linewidth=2, label=f"Average: {avg_gap:.2f}%")
    ax.legend(loc="lower right", fontsize=10)

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Instance gaps plot saved to: {output_path}")


def plot_instance_selection(
    selection_counts: dict[str, int],
    output_path: str | Path,
    title: str = "Instance Selection Distribution",
) -> None:
    """Plot how often each instance was selected during training.

    Args:
        selection_counts: Dict mapping instance_id to selection count.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not selection_counts:
        logger.warning("No selection data to plot")
        return

    # Sort by count (most selected first)
    sorted_counts = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)

    instances = [inst for inst, _ in sorted_counts]
    counts = [count for _, count in sorted_counts]

    total = sum(counts)
    percentages = [100 * c / total for c in counts]

    fig, ax = plt.subplots(figsize=(10, max(5, len(instances) * 0.4)))

    y_pos = range(len(instances))
    bars = ax.barh(y_pos, counts, color="#5C7AEA", edgecolor="black", linewidth=0.5)

    # Add percentage labels
    for bar, count, pct in zip(bars, counts, percentages):
        width = bar.get_width()
        ax.text(
            width + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(instances, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Selection Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add uniform line for reference
    uniform_count = total / len(instances)
    ax.axvline(x=uniform_count, color="red", linestyle="--", linewidth=2, label=f"Uniform: {uniform_count:.0f}")
    ax.legend(loc="lower right", fontsize=10)

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Instance selection plot saved to: {output_path}")


def plot_token_usage(
    budget_stats: dict,
    output_path: str | Path,
    model: str = "gpt-4o-mini",
) -> None:
    """Plot token usage breakdown (input vs output).

    Args:
        budget_stats: Dict with prompt_tokens, completion_tokens, max_tokens.
        output_path: Path to save the plot.
        model: Model name for cost estimation.
    """
    prompt_tokens = budget_stats.get("prompt_tokens", 0)
    completion_tokens = budget_stats.get("completion_tokens", 0)
    max_tokens = budget_stats.get("max_tokens", 500000)

    total_used = prompt_tokens + completion_tokens
    remaining = max(0, max_tokens - total_used)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart: token breakdown
    ax1 = axes[0]
    sizes = [prompt_tokens, completion_tokens, remaining]
    labels = [
        f"Input\n{prompt_tokens:,}",
        f"Output\n{completion_tokens:,}",
        f"Remaining\n{remaining:,}",
    ]
    colors = ["#3498db", "#e74c3c", "#95a5a6"]
    explode = (0.02, 0.02, 0)

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax1.set_title("Token Usage Breakdown", fontsize=14, fontweight="bold")

    # Bar chart: cost breakdown
    ax2 = axes[1]

    # Pricing per 1M tokens
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-5-mini": (0.30, 1.20),
        "gpt-5": (1.00, 8.00),
        "gpt-5.2": (1.75, 14.00),
    }
    input_price, output_price = pricing.get(model, (0.15, 0.60))

    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    categories = ["Input Tokens", "Output Tokens", "Total"]
    costs = [input_cost, output_cost, total_cost]
    bar_colors = ["#3498db", "#e74c3c", "#2ecc71"]

    bars = ax2.bar(categories, costs, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + total_cost * 0.02,
            f"${cost:.4f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax2.set_ylabel("Cost (USD)", fontsize=12, fontweight="bold")
    ax2.set_title(f"Estimated Cost ({model})", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, total_cost * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Token usage plot saved to: {output_path}")


def plot_synthesis_timeline(
    synthesis_events: list[tuple[float, str, bool]],
    convergence_history: list[tuple[float, float]],
    output_path: str | Path,
    title: str = "synthesized Synthesis Timeline",
) -> None:
    """Plot synthesis events on a timeline with convergence.

    Args:
        synthesis_events: List of (time, operator_name, success) tuples.
        convergence_history: List of (time, gap) tuples.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not synthesis_events and not convergence_history:
        logger.warning("No data to plot for synthesis timeline")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract times and gaps from convergence_history
    times = []
    gaps = []
    if convergence_history:
        # Handle both formats: list of dicts or list of tuples
        if isinstance(convergence_history[0], dict):
            times = [h.get("time", 0) for h in convergence_history]
            gaps = [h.get("value", 0) for h in convergence_history]
        else:
            times = [t for t, _ in convergence_history]
            gaps = [g for _, g in convergence_history]
        ax.plot(times, gaps, color="#2E86AB", linewidth=2, label="Avg Gap", zorder=2)
        ax.fill_between(times, gaps, alpha=0.2, color="#2E86AB")

    # Plot synthesis events
    success_times = []
    success_names = []
    fail_times = []
    fail_names = []

    for t, name, success in synthesis_events:
        if success:
            success_times.append(t)
            success_names.append(name)
        else:
            fail_times.append(t)
            fail_names.append(name)

    # Get gap values at synthesis times for y-position
    def get_gap_at_time(t):
        if not convergence_history:
            return 50  # Default
        # Use the already extracted times/gaps lists
        for time_val, gap_val in zip(times, gaps):
            if time_val >= t:
                return gap_val
        return gaps[-1] if gaps else 50

    # Plot successful syntheses
    if success_times:
        success_gaps = [get_gap_at_time(t) for t in success_times]
        ax.scatter(
            success_times,
            success_gaps,
            s=150,
            c="#27ae60",
            marker="^",
            edgecolors="black",
            linewidths=1,
            label=f"Synthesis Success ({len(success_times)})",
            zorder=3,
        )
        # Add vertical lines
        for t in success_times:
            ax.axvline(x=t, color="#27ae60", linestyle="--", alpha=0.3, linewidth=1)

    # Plot failed syntheses
    if fail_times:
        fail_gaps = [get_gap_at_time(t) for t in fail_times]
        ax.scatter(
            fail_times,
            fail_gaps,
            s=100,
            c="#e74c3c",
            marker="x",
            linewidths=2,
            label=f"Synthesis Failed ({len(fail_times)})",
            zorder=3,
        )

    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    if convergence_history:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Synthesis timeline plot saved to: {output_path}")


def visualize_best_path(
    role_path: list[str],
    operator_path: list[str],
    output_path: str | Path,
    title: str = "Best Path",
    gap: float | None = None,
    meta_graph: Any = None,
) -> None:
    """Visualize the best path found by ACO.

    Shows the metagraph with the best path highlighted.

    Args:
        role_path: List of role names in the path (e.g., ["const_nn", "ls_2opt", ...]).
        operator_path: List of operator names in the path.
        output_path: Path to save the visualization image.
        title: Title for the visualization.
        gap: Optional gap percentage to show in title.
        meta_graph: Optional MetaGraph for full edge information.
    """
    if not role_path:
        logger.warning("No path to visualize")
        return

    # Create directed graph
    G = nx.DiGraph()

    # Collect all roles (from path and metagraph)
    all_roles = set(role_path)
    if meta_graph is not None:
        for (src_role, tgt_role) in meta_graph.edges.keys():
            src = src_role.value if hasattr(src_role, "value") else str(src_role)
            tgt = tgt_role.value if hasattr(tgt_role, "value") else str(tgt_role)
            all_roles.add(src)
            all_roles.add(tgt)

    for role in all_roles:
        G.add_node(role)

    # Add all edges from metagraph (gray, thin)
    if meta_graph is not None:
        for (src_role, tgt_role), edge in meta_graph.edges.items():
            src = src_role.value if hasattr(src_role, "value") else str(src_role)
            tgt = tgt_role.value if hasattr(tgt_role, "value") else str(tgt_role)
            G.add_edge(src, tgt, in_path=False, weight=edge.weight)

    # Mark path edges
    path_edges = []
    for i in range(len(role_path) - 1):
        src, tgt = role_path[i], role_path[i + 1]
        path_edges.append((src, tgt))
        if G.has_edge(src, tgt):
            G[src][tgt]["in_path"] = True
        else:
            G.add_edge(src, tgt, in_path=True, weight=1.0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Layout - organize by role category
    pos = {}
    role_list = list(all_roles)
    const_roles = sorted([r for r in role_list if r.startswith("const")])
    ls_roles = sorted([r for r in role_list if r.startswith("ls")])
    pert_roles = sorted([r for r in role_list if r.startswith("pert")])
    meta_roles = sorted([r for r in role_list if r.startswith("meta")])

    y_positions = {"const": 3, "ls": 1, "pert": -1, "meta": -3}

    def place_roles(role_list: list, y: float, x_start: float = 0) -> None:
        n = len(role_list)
        x_spacing = 2.5
        x_offset = -(n - 1) * x_spacing / 2 + x_start
        for i, role in enumerate(role_list):
            pos[role] = (x_offset + i * x_spacing, y)

    place_roles(const_roles, y_positions["const"])
    place_roles(ls_roles, y_positions["ls"])
    place_roles(pert_roles, y_positions["pert"], x_start=-1)
    place_roles(meta_roles, y_positions["meta"], x_start=1)

    # Draw non-path edges first (gray, thin)
    non_path_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("in_path", False)]
    if non_path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=non_path_edges,
            width=0.5,
            alpha=0.2,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

    # Draw path edges (colored, thick)
    if path_edges:
        # Count edge usage to handle repeated edges
        edge_counts: dict[tuple[str, str], int] = {}
        for src, tgt in path_edges:
            key = (src, tgt)
            edge_counts[key] = edge_counts.get(key, 0) + 1

        # Use a color gradient for the path
        n_edges = len(path_edges)
        colors = plt.cm.viridis(range(0, 256, 256 // max(n_edges, 1)))[:n_edges]

        # Track which edges we've drawn to vary curvature for repeated edges
        drawn_edges: dict[tuple[str, str], int] = {}

        for idx, (src, tgt) in enumerate(path_edges):
            key = (src, tgt)
            times_drawn = drawn_edges.get(key, 0)
            drawn_edges[key] = times_drawn + 1

            # Vary curvature for repeated edges and bidirectional edges
            base_rad = 0.15
            if edge_counts[key] > 1:
                # Multiple uses of same edge: vary curvature
                rad = base_rad + times_drawn * 0.15
            elif G.has_edge(tgt, src):
                # Bidirectional edge
                rad = base_rad if src < tgt else -base_rad
            else:
                rad = base_rad

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(src, tgt)],
                width=3.0,
                alpha=0.9,
                edge_color=[colors[idx]],
                arrows=True,
                arrowsize=20,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                min_source_margin=25,
                min_target_margin=25,
                ax=ax,
            )

    # Node colors: highlight nodes in path
    path_role_set = set(role_path)
    node_colors = []
    node_edge_colors = []
    node_sizes = []

    for role in G.nodes():
        in_path = role in path_role_set
        if role.startswith("const"):
            color = "#0CF574" if in_path else "#A0F5D8"
        elif role.startswith("ls"):
            color = "#2F97C1" if in_path else "#97CBE0"
        elif role.startswith("pert"):
            color = "#1CCAD8" if in_path else "#8DE4EB"
        else:
            color = "#15E6CD" if in_path else "#8AF2E6"

        node_colors.append(color)
        node_edge_colors.append("black" if in_path else "gray")
        node_sizes.append(3500 if in_path else 2000)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=[2.5 if role in path_role_set else 1 for role in G.nodes()],
        ax=ax,
    )

    # Node labels
    labels = {role: role.replace("_", "\n") for role in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    # Build path sequence text for the side panel
    from src.geakg.layers.l1.base_operators import ALL_ROLES
    base_op_names = {f"{role}_base" for role in ALL_ROLES}

    path_lines = []
    for idx, (role, op) in enumerate(zip(role_path, operator_path)):
        is_generated = op not in base_op_names
        op_short = op[:25] + ".." if len(op) > 25 else op
        marker = "★" if is_generated else "  "
        path_lines.append(f"{idx + 1:2d}. {marker} {role} → {op_short}")

    # Add path sequence as text box on the right
    path_text = "PATH SEQUENCE:\n" + "\n".join(path_lines)
    ax.text(
        1.02, 0.98, path_text,
        transform=ax.transAxes,
        fontsize=7,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray"),
    )

    # Legend (simplified, on upper left)
    legend_elements = [
        mpatches.Patch(color="#0CF574", label="Construction"),
        mpatches.Patch(color="#2F97C1", label="Local Search"),
        mpatches.Patch(color="#1CCAD8", label="Perturbation"),
        mpatches.Patch(color="#A0F5D8", edgecolor="gray", label="Not in path"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Title with gap
    full_title = title
    if gap is not None:
        full_title += f" (Gap: {gap:.2f}%)"

    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=10)

    ax.set_xlim(-10, 8)
    ax.set_ylim(-5, 5)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Best path visualization saved to: {output_path}")


def plot_all_multi_instance(
    result: dict,
    output_dir: str | Path,
    model: str = "gpt-4o-mini",
) -> None:
    """Generate all plots for multi-instance experiment.

    Args:
        result: Full experiment result dict.
        output_dir: Directory to save plots.
        model: Model name for cost estimation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_instances = result.get("n_instances", 0)

    # 1. Convergence plot
    plot_convergence(
        result,
        output_dir / "convergence.png",
        multi_instance=True,
        n_instances=n_instances,
    )

    # 2. Per-instance gaps
    per_instance_gaps = result.get("per_instance_gaps", {})
    if per_instance_gaps:
        plot_instance_gaps(
            per_instance_gaps,
            output_dir / "instance_gaps.png",
            title=f"Final Gap per Instance (Champion Algorithm)",
        )

    # 3. Instance selection distribution
    selection_stats = result.get("instance_selection_stats", {})
    if selection_stats:
        plot_instance_selection(
            selection_stats,
            output_dir / "instance_selection.png",
            title="Instance Hardness Sampling Distribution",
        )

    # 4. Token usage
    budget_stats = result.get("budget", {})
    if budget_stats:
        plot_token_usage(
            budget_stats,
            output_dir / "token_usage.png",
            model=model,
        )

    # 5. Synthesis timeline (if we have synthesis events)
    convergence_history = result.get("convergence_history", [])
    synth_appearances_raw = result.get("synth_first_appearance", [])

    # Convert synth_appearances to synthesis events format
    synthesis_events = []
    if isinstance(synth_appearances_raw, list):
        # Multi-instance format: list of dicts
        for item in synth_appearances_raw:
            t = item.get("time", 0)
            op_id = item.get("operator_id", "unknown")
            synthesis_events.append((t, op_id, True))
    elif isinstance(synth_appearances_raw, dict):
        # Single-instance format: dict
        for op_id, (t, _, _) in synth_appearances_raw.items():
            synthesis_events.append((t, op_id, True))

    if synthesis_events or convergence_history:
        plot_synthesis_timeline(
            synthesis_events,
            convergence_history,
            output_dir / "synthesis_timeline.png",
            title="synthesized Synthesis Events on Convergence Curve",
        )

    logger.info(f"All multi-instance plots saved to: {output_dir}")
