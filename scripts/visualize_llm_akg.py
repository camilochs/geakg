"""Generate AKG with LLM-provided compatible pairs and visualize."""
import sys
sys.path.insert(0, "/Users/cchac/Workspace/01_Me/Research/Algorithmic_Knowledge_Graphs")

from src.geakg.generator import LLMAKGGenerator, OperatorInfo
from src.llm.client import OllamaClient
from src.llm.config import LLMConfig


def get_tsp_operators() -> list[OperatorInfo]:
    """Get TSP operators."""
    return [
        # Construction
        OperatorInfo(id="greedy_nearest_neighbor", name="Greedy Nearest Neighbor",
                    description="Builds tour by always visiting closest unvisited city"),
        OperatorInfo(id="farthest_insertion", name="Farthest Insertion",
                    description="Inserts farthest city into partial tour at best position"),
        OperatorInfo(id="cheapest_insertion", name="Cheapest Insertion",
                    description="Inserts city that minimizes tour length increase"),
        OperatorInfo(id="random_insertion", name="Random Insertion",
                    description="Inserts random city at best position in partial tour"),
        OperatorInfo(id="savings_algorithm", name="Savings Algorithm",
                    description="Merges routes by maximum savings"),
        OperatorInfo(id="christofides_init", name="Christofides Initialization",
                    description="Uses minimum spanning tree as base for initial tour"),

        # Local Search
        OperatorInfo(id="two_opt", name="2-opt",
                    description="Reverses segment between two edges to remove crossings"),
        OperatorInfo(id="three_opt", name="3-opt",
                    description="Reconnects three broken edges in best configuration"),
        OperatorInfo(id="or_opt", name="Or-opt",
                    description="Relocates sequences of 1-3 cities to better positions"),
        OperatorInfo(id="node_insertion", name="Node Insertion",
                    description="Moves single node to best position in tour"),
        OperatorInfo(id="node_swap", name="Node Swap",
                    description="Swaps positions of two nodes in tour"),
        OperatorInfo(id="lk_moves", name="Lin-Kernighan Moves",
                    description="Variable-depth edge exchanges for improvement"),
        OperatorInfo(id="segment_shift", name="Segment Shift",
                    description="Shifts a segment of tour to different position"),

        # Perturbation
        OperatorInfo(id="double_bridge", name="Double Bridge",
                    description="Splits tour into 4 parts and reconnects differently"),
        OperatorInfo(id="random_segment_reverse", name="Random Segment Reverse",
                    description="Reverses random segment of tour"),
        OperatorInfo(id="random_four_swap", name="Random 4-Swap",
                    description="Swaps four random pairs of cities"),
        OperatorInfo(id="guided_perturbation", name="Guided Perturbation",
                    description="Perturbs based on edge frequencies in good solutions"),
        OperatorInfo(id="kick_move", name="Kick Move",
                    description="Large perturbation to escape deep local optima"),

        # Meta-heuristic
        OperatorInfo(id="simulated_annealing_step", name="SA Step",
                    description="Accept worse solutions with temperature-based probability"),
        OperatorInfo(id="tabu_check", name="Tabu Check",
                    description="Prevents recently used moves for diversification"),
        OperatorInfo(id="adaptive_restart", name="Adaptive Restart",
                    description="Restarts search with learned parameters"),
        OperatorInfo(id="population_crossover", name="Population Crossover",
                    description="Combines solutions from population"),
    ]


def visualize_akg(akg, output_path="akg_llm_pairs.png"):
    """Visualize AKG with networkx."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    # Color map for categories
    color_map = {
        "CONSTRUCTION": "#90EE90",     # Light green
        "LOCAL_SEARCH": "#87CEEB",     # Light blue
        "PERTURBATION": "#FFB6C1",     # Light pink
        "META_HEURISTIC": "#DDA0DD",   # Plum
    }

    # Add nodes
    for node_id, node in akg.nodes.items():
        G.add_node(node_id, category=node.category.name)

    # Add edges
    for edge in akg.edges.values():
        G.add_edge(edge.source, edge.target, weight=edge.weight)

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    plt.figure(figsize=(16, 12))

    # Draw nodes by category
    for cat_name, color in color_map.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("category") == cat_name]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                                   node_size=1200, alpha=0.9)

    # Draw edges with varying thickness based on weight
    edges = G.edges(data=True)
    weights = [d.get("weight", 0.5) for _, _, d in edges]
    max_w = max(weights) if weights else 1
    widths = [1.0 + 3.0 * (w / max_w) for w in weights]

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5,
                          edge_color="gray", arrows=True,
                          arrowsize=15, connectionstyle="arc3,rad=0.1")

    # Labels
    labels = {n: n.replace("_", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    # Legend
    legend_elements = [plt.scatter([], [], c=color, s=100, label=cat.replace("_", " ").title())
                       for cat, color in color_map.items()]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.title(f"AKG with LLM-Provided Compatible Pairs\n{len(G.nodes())} nodes, {len(G.edges())} edges",
             fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return G


def main():
    # Create LLM client - use 7b for better quality
    config = LLMConfig(model="qwen2.5:7b", cache_enabled=False)
    client = OllamaClient(config=config)

    # Get operators
    operators = get_tsp_operators()
    print(f"Operators: {len(operators)}")

    # Generate AKG with more connectivity iterations
    generator = LLMAKGGenerator(client, operators)
    akg = generator.generate(max_connectivity_iterations=5)

    if akg is None:
        print("Failed to generate AKG")
        return

    print(f"\nAKG generated: {len(akg.nodes)} nodes, {len(akg.edges)} edges")

    # Show edge statistics
    edges_by_source = {}
    for edge in akg.edges.values():
        if edge.source not in edges_by_source:
            edges_by_source[edge.source] = []
        edges_by_source[edge.source].append((edge.target, edge.weight))

    print("\n=== Edge structure ===")
    for src, targets in sorted(edges_by_source.items()):
        targets_str = ", ".join([f"{t}({w:.2f})" for t, w in sorted(targets, key=lambda x: -x[1])])
        print(f"{src[:20]:20s} -> {targets_str}")

    # Visualize
    G = visualize_akg(akg, "akg_llm_pairs.png")

    # Check graph properties
    print(f"\n=== Graph properties ===")
    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")

    # Count by category
    from collections import Counter
    categories = Counter(d["category"] for _, d in G.nodes(data=True))
    print(f"By category: {dict(categories)}")

    # Check connectivity from construction
    constructions = [n for n, d in G.nodes(data=True) if d["category"] == "CONSTRUCTION"]
    print(f"\nConstruction nodes: {constructions}")

    import networkx as nx
    for c in constructions:
        reachable = len(nx.descendants(G, c))
        print(f"  {c}: reaches {reachable} nodes")


if __name__ == "__main__":
    main()
