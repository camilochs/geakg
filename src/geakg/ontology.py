"""Initial ontology for the Algorithmic Knowledge Graph.

Defines the 30 initial operator nodes and their relationships.
These form the foundation of the algorithm design space.
"""

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import (
    AKGEdge,
    DataStructureNode,
    EdgeType,
    OperatorCategory,
    OperatorNode,
    PropertyNode,
)


def create_construction_operators() -> list[OperatorNode]:
    """Create construction operator nodes (10 operators)."""
    return [
        OperatorNode(
            id="greedy_nearest_neighbor",
            name="Greedy Nearest Neighbor",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by always visiting the nearest unvisited city",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={"start_node": "random"},
        ),
        OperatorNode(
            id="farthest_insertion",
            name="Farthest Insertion",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by inserting the farthest city from current tour",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={},
        ),
        OperatorNode(
            id="cheapest_insertion",
            name="Cheapest Insertion",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by inserting city at cheapest position",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={},
        ),
        OperatorNode(
            id="random_insertion",
            name="Random Insertion",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by inserting cities in random order",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={"seed": None},
        ),
        OperatorNode(
            id="savings_heuristic",
            name="Clarke-Wright Savings",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour using savings algorithm (merging routes)",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={"lambda": 1.0},
            domains=["tsp", "vrp"],
        ),
        OperatorNode(
            id="christofides_construction",
            name="Christofides Construction",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour using MST + matching (1.5 approximation)",
            preconditions=[],
            effects=["has_solution", "has_tour", "has_bound"],
            parameters={},
            domains=["tsp"],
        ),
        OperatorNode(
            id="nearest_addition",
            name="Nearest Addition",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by adding nearest city to tour",
            preconditions=[],
            effects=["has_solution", "has_tour"],
            parameters={},
        ),
        OperatorNode(
            id="convex_hull_start",
            name="Convex Hull Initialization",
            category=OperatorCategory.CONSTRUCTION,
            description="Start with convex hull, then insert remaining cities",
            preconditions=["has_coordinates"],
            effects=["has_solution", "has_tour"],
            parameters={},
            domains=["tsp"],
        ),
        OperatorNode(
            id="cluster_first",
            name="Cluster First Route Second",
            category=OperatorCategory.CONSTRUCTION,
            description="Cluster cities then route within clusters",
            preconditions=[],
            effects=["has_solution", "has_tour", "has_clusters"],
            parameters={"n_clusters": "auto"},
            domains=["tsp", "vrp"],
        ),
        OperatorNode(
            id="sweep_algorithm",
            name="Sweep Algorithm",
            category=OperatorCategory.CONSTRUCTION,
            description="Build tour by sweeping around depot angle",
            preconditions=["has_coordinates"],
            effects=["has_solution", "has_tour"],
            parameters={"start_angle": 0.0},
            domains=["tsp", "vrp"],
        ),
    ]


def create_local_search_operators() -> list[OperatorNode]:
    """Create local search operator nodes (8 operators)."""
    return [
        OperatorNode(
            id="two_opt",
            name="2-opt",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Remove two edges and reconnect in the other way",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour"],
            parameters={"max_iterations": 1000, "first_improvement": False},
        ),
        OperatorNode(
            id="three_opt",
            name="3-opt",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Remove three edges and reconnect optimally",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour"],
            parameters={"max_iterations": 500},
        ),
        OperatorNode(
            id="or_opt",
            name="Or-opt",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Relocate segments of 1, 2, or 3 consecutive cities",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour"],
            parameters={"segment_sizes": [1, 2, 3]},
        ),
        OperatorNode(
            id="swap",
            name="Swap",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Swap positions of two cities in the tour",
            preconditions=["has_tour"],
            effects=["has_solution"],
            parameters={},
        ),
        OperatorNode(
            id="insert",
            name="Insert/Relocate",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Remove a city and insert at best position",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour"],
            parameters={},
        ),
        OperatorNode(
            id="invert",
            name="Invert/Reverse",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Reverse a segment of the tour",
            preconditions=["has_tour"],
            effects=["has_solution"],
            parameters={"segment_length": "variable"},
        ),
        OperatorNode(
            id="lin_kernighan",
            name="Lin-Kernighan",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Variable-depth search with sequential edge exchanges",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour", "near_optimal"],
            parameters={"max_depth": 5, "backtracking": True},
        ),
        OperatorNode(
            id="variable_neighborhood",
            name="Variable Neighborhood Descent",
            category=OperatorCategory.LOCAL_SEARCH,
            description="Systematically explore multiple neighborhood structures",
            preconditions=["has_tour"],
            effects=["has_solution", "improved_tour"],
            parameters={"neighborhoods": ["swap", "insert", "two_opt"]},
        ),
    ]


def create_perturbation_operators() -> list[OperatorNode]:
    """Create perturbation operator nodes (6 operators)."""
    return [
        OperatorNode(
            id="double_bridge",
            name="Double Bridge",
            category=OperatorCategory.PERTURBATION,
            description="Remove 4 edges forming two bridges, reconnect differently",
            preconditions=["has_tour"],
            effects=["has_solution", "escaped_local_opt"],
            parameters={},
        ),
        OperatorNode(
            id="random_segment_shuffle",
            name="Random Segment Shuffle",
            category=OperatorCategory.PERTURBATION,
            description="Divide tour into segments and shuffle their order",
            preconditions=["has_tour"],
            effects=["has_solution", "escaped_local_opt"],
            parameters={"n_segments": 4},
        ),
        OperatorNode(
            id="guided_mutation",
            name="Guided Mutation",
            category=OperatorCategory.PERTURBATION,
            description="Mutate based on edge frequency in good solutions",
            preconditions=["has_tour", "has_history"],
            effects=["has_solution", "diversified"],
            parameters={"mutation_strength": 0.3},
        ),
        OperatorNode(
            id="ruin_recreate",
            name="Ruin and Recreate",
            category=OperatorCategory.PERTURBATION,
            description="Remove portion of solution and rebuild",
            preconditions=["has_tour"],
            effects=["has_solution", "escaped_local_opt"],
            parameters={"ruin_fraction": 0.3, "recreate_method": "greedy"},
        ),
        OperatorNode(
            id="large_neighborhood_search",
            name="Large Neighborhood Search",
            category=OperatorCategory.PERTURBATION,
            description="Destroy and repair using problem-specific operators",
            preconditions=["has_tour"],
            effects=["has_solution", "escaped_local_opt", "diversified"],
            parameters={"destroy_operators": [], "repair_operators": []},
        ),
        OperatorNode(
            id="adaptive_mutation",
            name="Adaptive Mutation",
            category=OperatorCategory.PERTURBATION,
            description="Mutation with self-adjusting strength",
            preconditions=["has_tour"],
            effects=["has_solution"],
            parameters={"initial_rate": 0.1, "adaptation": "success_rate"},
        ),
    ]


def create_data_structure_nodes() -> list[DataStructureNode]:
    """Create data structure nodes (supporting infrastructure)."""
    return [
        DataStructureNode(
            id="adjacency_matrix",
            name="Adjacency Matrix",
            description="Distance matrix for O(1) edge lookup",
            memory_complexity="O(n^2)",
            supported_operations=["get_distance", "get_neighbors"],
        ),
        DataStructureNode(
            id="candidate_list",
            name="Candidate List",
            description="Pre-computed nearest neighbors for each city",
            memory_complexity="O(n*k)",
            supported_operations=["get_nearest_k", "restrict_neighbors"],
        ),
        DataStructureNode(
            id="tabu_list",
            name="Tabu List",
            description="Recently visited moves to avoid",
            memory_complexity="O(tenure)",
            supported_operations=["add", "contains", "expire"],
        ),
        DataStructureNode(
            id="priority_queue",
            name="Priority Queue",
            description="Heap for ordering moves by improvement",
            memory_complexity="O(n)",
            supported_operations=["push", "pop_best", "update"],
        ),
    ]


def create_property_nodes() -> list[PropertyNode]:
    """Create property nodes (problem characteristics)."""
    return [
        PropertyNode(
            id="problem_size",
            name="Problem Size",
            description="Number of cities/nodes in the problem",
            value_type="numeric",
            range=(2, 100000),
        ),
        PropertyNode(
            id="distance_metric",
            name="Distance Metric",
            description="Type of distance function used",
            value_type="categorical",
        ),
        PropertyNode(
            id="symmetry",
            name="Symmetry",
            description="Whether distances are symmetric",
            value_type="boolean",
        ),
        PropertyNode(
            id="clustering_coefficient",
            name="Clustering Coefficient",
            description="Degree of spatial clustering in cities",
            value_type="numeric",
            range=(0.0, 1.0),
        ),
    ]


def create_edges(akg: AlgorithmicKnowledgeGraph) -> None:
    """Create edges defining valid transitions and relationships.

    Uses 3 categories: construction, local_search, perturbation.
    """
    # Get all operator IDs by category
    construction_ops = [n.id for n in akg.get_operators_by_category(OperatorCategory.CONSTRUCTION)]
    local_search_ops = [n.id for n in akg.get_operators_by_category(OperatorCategory.LOCAL_SEARCH)]
    perturbation_ops = [n.id for n in akg.get_operators_by_category(OperatorCategory.PERTURBATION)]

    # Construction -> Local Search (high weight, common pattern)
    for constr in construction_ops:
        for local in local_search_ops:
            akg.add_edge(
                AKGEdge(
                    source=constr,
                    target=local,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=0.8,
                )
            )

    # Local Search -> Local Search (medium weight, can chain different searches)
    for ls1 in local_search_ops:
        for ls2 in local_search_ops:
            if ls1 != ls2:
                akg.add_edge(
                    AKGEdge(
                        source=ls1,
                        target=ls2,
                        edge_type=EdgeType.SEQUENTIAL,
                        weight=0.5,
                    )
                )

    # Local Search -> Perturbation (medium-high weight, escape local optima)
    for local in local_search_ops:
        for pert in perturbation_ops:
            akg.add_edge(
                AKGEdge(
                    source=local,
                    target=pert,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=0.7,
                )
            )

    # Perturbation -> Local Search (high weight, refine after perturbation)
    for pert in perturbation_ops:
        for local in local_search_ops:
            akg.add_edge(
                AKGEdge(
                    source=pert,
                    target=local,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=0.85,
                )
            )

    # Specific high-value edges (expert knowledge)
    high_value_edges = [
        ("greedy_nearest_neighbor", "two_opt", 0.95),  # Classic combination
        ("two_opt", "three_opt", 0.7),  # Progressive refinement
        ("two_opt", "double_bridge", 0.8),  # ILS pattern
        ("double_bridge", "two_opt", 0.9),  # Return from perturbation
        ("christofides_construction", "two_opt", 0.85),  # Near-optimal start
        ("lin_kernighan", "double_bridge", 0.75),  # Powerful + escape
        ("cluster_first", "two_opt", 0.8),  # Structured approach
        ("ruin_recreate", "lin_kernighan", 0.85),  # Strong combination
    ]

    for source, target, weight in high_value_edges:
        # Update weight if edge exists, otherwise add new edge
        if (source, target) in akg.edges:
            akg.update_edge_weight(source, target, weight - akg.edges[(source, target)].weight)
        else:
            akg.add_edge(
                AKGEdge(
                    source=source,
                    target=target,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=weight,
                )
            )


def create_default_akg() -> AlgorithmicKnowledgeGraph:
    """Create the default AKG with 30 operator nodes.

    Returns:
        Initialized AKG with operators and edges
    """
    akg = AlgorithmicKnowledgeGraph()

    # Add all operator nodes
    for node in create_construction_operators():
        akg.add_node(node)

    for node in create_local_search_operators():
        akg.add_node(node)

    for node in create_perturbation_operators():
        akg.add_node(node)

    # Add data structure nodes
    for node in create_data_structure_nodes():
        akg.add_node(node)

    # Add property nodes
    for node in create_property_nodes():
        akg.add_node(node)

    # Create edges between operators
    create_edges(akg)

    return akg


# Convenience function for quick access
def get_operator_summary() -> dict[str, list[str]]:
    """Get summary of operators by category.

    Returns:
        Dict mapping category to list of operator IDs
    """
    return {
        "construction": [
            "greedy_nearest_neighbor",
            "farthest_insertion",
            "cheapest_insertion",
            "random_insertion",
            "savings_heuristic",
            "christofides_construction",
            "nearest_addition",
            "convex_hull_start",
            "cluster_first",
            "sweep_algorithm",
        ],
        "local_search": [
            "two_opt",
            "three_opt",
            "or_opt",
            "swap",
            "insert",
            "invert",
            "lin_kernighan",
            "variable_neighborhood",
        ],
        "perturbation": [
            "double_bridge",
            "random_segment_shuffle",
            "guided_mutation",
            "ruin_recreate",
            "large_neighborhood_search",
            "adaptive_mutation",
        ],
    }
