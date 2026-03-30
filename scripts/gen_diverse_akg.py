"""Generate AKG with strict diversity validation."""
import sys
sys.path.insert(0, "/Users/cchac/Workspace/01_Me/Research/Algorithmic_Knowledge_Graphs")

from src.geakg.generator import LLMAKGGenerator, OperatorInfo
from src.llm.client import OllamaClient
from src.llm.config import LLMConfig


def get_tsp_operators() -> list[OperatorInfo]:
    """Get TSP operators with correct categories."""
    return [
        # Construction
        OperatorInfo(id="greedy_nearest_neighbor", name="Greedy Nearest Neighbor",
                    description="Builds tour by always visiting closest unvisited city",
                    category="construction"),
        OperatorInfo(id="farthest_insertion", name="Farthest Insertion",
                    description="Inserts farthest city into partial tour at best position",
                    category="construction"),
        OperatorInfo(id="cheapest_insertion", name="Cheapest Insertion",
                    description="Inserts city that minimizes tour length increase",
                    category="construction"),
        OperatorInfo(id="random_insertion", name="Random Insertion",
                    description="Inserts random city at best position in partial tour",
                    category="construction"),
        OperatorInfo(id="savings_heuristic", name="Savings Heuristic",
                    description="Merges routes by maximum savings",
                    category="construction"),
        OperatorInfo(id="convex_hull_start", name="Convex Hull Start",
                    description="Starts tour from convex hull of cities",
                    category="construction"),

        # Local Search (7 operators)
        OperatorInfo(id="two_opt", name="2-opt",
                    description="Reverses segment between two edges to remove crossings",
                    category="local_search"),
        OperatorInfo(id="three_opt", name="3-opt",
                    description="Reconnects three broken edges in best configuration",
                    category="local_search"),
        OperatorInfo(id="or_opt", name="Or-opt",
                    description="Relocates sequences of 1-3 cities to better positions",
                    category="local_search"),
        OperatorInfo(id="swap", name="Swap",
                    description="Swaps positions of two cities in tour",
                    category="local_search"),
        OperatorInfo(id="insert", name="Insert",
                    description="Removes and reinserts city at best position",
                    category="local_search"),
        OperatorInfo(id="lin_kernighan", name="Lin-Kernighan",
                    description="Variable-depth edge exchanges",
                    category="local_search"),
        OperatorInfo(id="variable_neighborhood", name="Variable Neighborhood Descent",
                    description="Cycles through multiple neighborhood structures",
                    category="local_search"),

        # Perturbation (3 operators)
        OperatorInfo(id="double_bridge", name="Double Bridge",
                    description="Splits tour into 4 parts and reconnects differently",
                    category="perturbation"),
        OperatorInfo(id="random_segment_shuffle", name="Random Segment Shuffle",
                    description="Shuffles segments of the tour",
                    category="perturbation"),
        OperatorInfo(id="ruin_recreate", name="Ruin and Recreate",
                    description="Removes part of solution and rebuilds",
                    category="perturbation"),

        # Meta-heuristic (3 operators)
        OperatorInfo(id="simulated_annealing_step", name="SA Step",
                    description="Accept worse solutions with temperature-based probability",
                    category="meta_heuristic"),
        OperatorInfo(id="tabu_search_step", name="Tabu Step",
                    description="Prevents recently used moves for diversification",
                    category="meta_heuristic"),
        OperatorInfo(id="iterated_local_search", name="ILS Step",
                    description="Perturb and re-optimize cycle",
                    category="meta_heuristic"),
    ]


def visualize_akg(akg, filename: str):
    """Visualize AKG with diversity analysis using layered layout."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from src.geakg.nodes import OperatorCategory

    G = nx.DiGraph()

    # Add nodes with categories
    for node_id, node in akg.nodes.items():
        G.add_node(node_id, category=node.category.value)

    # Add edges
    for (src, tgt), edge in akg.edges.items():
        G.add_edge(src, tgt, weight=edge.weight)

    # Color by category
    color_map = {
        "construction": "#90EE90",
        "local_search": "#87CEEB",
        "perturbation": "#FFB6C1",
        "meta_heuristic": "#DDA0DD",
    }

    node_colors = [color_map.get(G.nodes[n].get("category", ""), "#GRAY") for n in G.nodes()]

    # Layered layout by category (left to right)
    # Layer 0: Construction, Layer 1: Local Search, Layer 2: Perturbation, Layer 3: Meta-heuristic
    layer_x = {
        "construction": 0,
        "local_search": 1,
        "perturbation": 2,
        "meta_heuristic": 3,
    }

    # Group nodes by category
    nodes_by_category = {}
    for n, data in G.nodes(data=True):
        cat = data.get("category", "")
        if cat not in nodes_by_category:
            nodes_by_category[cat] = []
        nodes_by_category[cat].append(n)

    # Assign positions
    pos = {}
    for cat, nodes in nodes_by_category.items():
        x = layer_x.get(cat, 0)
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            # Spread vertically, centered
            y = (i - (n_nodes - 1) / 2) * 1.2
            pos[node] = (x * 3, y)

    plt.figure(figsize=(16, 12))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.9)

    # Draw labels
    labels = {n: n.replace("_", "\n")[:12] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Count escape and re-optimize edges
    escape_edges = []
    reopt_edges = []
    other_edges = []

    ls_ops = {n for n, d in G.nodes(data=True) if d.get("category") == "local_search"}
    pert_ops = {n for n, d in G.nodes(data=True) if d.get("category") == "perturbation"}

    for u, v in G.edges():
        if u in ls_ops and v in pert_ops:
            escape_edges.append((u, v))
        elif u in pert_ops and v in ls_ops:
            reopt_edges.append((u, v))
        else:
            other_edges.append((u, v))

    # Draw edges with colors
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, alpha=0.5,
                          edge_color="gray", arrows=True, arrowsize=15,
                          connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=escape_edges, alpha=0.8,
                          edge_color="red", width=2, arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=reopt_edges, alpha=0.8,
                          edge_color="cyan", width=2, arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', markersize=10, label='Construction'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=10, label='Local Search'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB6C1', markersize=10, label='Perturbation'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DDA0DD', markersize=10, label='Meta-heuristic'),
        plt.Line2D([0], [0], color='red', linewidth=2, label=f'ESCAPE ({len(escape_edges)} edges)'),
        plt.Line2D([0], [0], color='cyan', linewidth=2, label=f'RE-OPT ({len(reopt_edges)} edges)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # Add layer labels at the top
    ax = plt.gca()
    layer_labels = ["Construction", "Local Search", "Perturbation", "Meta-heuristic"]
    for i, label in enumerate(layer_labels):
        ax.text(i * 3, max(y for x, y in pos.values()) + 1.0, label,
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               color=list(color_map.values())[i])

    # Diversity analysis
    ls_with_escape = {u for u, v in escape_edges}
    ls_without = ls_ops - ls_with_escape
    pert_with_reopt = {u for u, v in reopt_edges}
    pert_without = pert_ops - pert_with_reopt

    title = f"AKG with Strict Diversity\n{len(G.nodes())} nodes, {len(G.edges())} edges"
    if ls_without:
        title += f"\nLS without escape: {sorted(ls_without)}"
    if pert_without:
        title += f"\nPert without re-opt: {sorted(pert_without)}"
    if not ls_without and not pert_without:
        title += "\n✓ ALL operators have required edges!"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDiversity Analysis:")
    print(f"  Local search operators: {len(ls_ops)}")
    print(f"  - With escape edge: {len(ls_with_escape)}")
    print(f"  - Without escape: {ls_without if ls_without else 'NONE ✓'}")
    print(f"  Perturbation operators: {len(pert_ops)}")
    print(f"  - With re-opt edge: {len(pert_with_reopt)}")
    print(f"  - Without re-opt: {pert_without if pert_without else 'NONE ✓'}")
    print(f"\nSaved: {filename}")


def main():
    print("=" * 60)
    print("Generating AKG with Strict Diversity Validation")
    print("=" * 60)

    config = LLMConfig(model="qwen2.5:7b", cache_enabled=False)
    client = OllamaClient(config=config)

    operators = get_tsp_operators()
    print(f"\nOperators: {len(operators)}")
    print(f"  Construction: {sum(1 for o in operators if o.category == 'construction')}")
    print(f"  Local Search: {sum(1 for o in operators if o.category == 'local_search')}")
    print(f"  Perturbation: {sum(1 for o in operators if o.category == 'perturbation')}")
    print(f"  Meta-heuristic: {sum(1 for o in operators if o.category == 'meta_heuristic')}")

    generator = LLMAKGGenerator(client, operators, max_retries=5)

    print("\nGenerating AKG (strict diversity validation)...")
    akg = generator.generate(max_connectivity_iterations=5)

    if akg is None:
        print("ERROR: Failed to generate valid AKG")
        return

    print(f"\nAKG Generated: {len(akg.nodes)} nodes, {len(akg.edges)} edges")

    # Show weight distribution by transition type
    print("\nWeight distribution by transition type:")
    weights_by_type = {}
    for edge in akg.edges.values():
        src_cat = next((o.category for o in operators if o.id == edge.source), "?")
        tgt_cat = next((o.category for o in operators if o.id == edge.target), "?")
        trans_type = f"{src_cat[:4]}→{tgt_cat[:4]}"
        if trans_type not in weights_by_type:
            weights_by_type[trans_type] = []
        weights_by_type[trans_type].append(edge.weight)

    for trans_type, weights in sorted(weights_by_type.items()):
        avg = sum(weights) / len(weights)
        print(f"  {trans_type}: {avg:.2f} ({len(weights)} edges)")

    visualize_akg(akg, "akg_strict_diversity.png")


if __name__ == "__main__":
    main()
