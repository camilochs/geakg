"""TSP Operators - Real implementations of all 30 AKG operators.

Usage:
    from src.operators.registry import OperatorRegistry, create_context

    # Create registry
    registry = OperatorRegistry()

    # Create context
    context = create_context(distance_matrix, coordinates, tour)

    # Execute operator with adaptive parameters
    result = registry.execute("two_opt", context)
    improved_tour = result.tour
"""

from src.operators.registry import (
    OperatorRegistry,
    ExecutionContext,
    OperatorResult,
    OperatorCategory,
    create_context,
)

from src.operators.construction import (
    greedy_nearest_neighbor,
    farthest_insertion,
    cheapest_insertion,
    random_insertion,
    savings_heuristic,
    christofides_construction,
    nearest_addition,
    convex_hull_start,
    cluster_first,
    sweep_algorithm,
)

from src.operators.local_search import (
    two_opt,
    three_opt,
    or_opt,
    swap_operator,
    insert_operator,
    invert_operator,
    lin_kernighan,
    variable_neighborhood_descent,
)

from src.operators.perturbation import (
    double_bridge,
    random_segment_shuffle,
    guided_mutation,
    ruin_recreate,
    large_neighborhood_search,
    adaptive_mutation,
)

from src.operators.meta_heuristic import (
    simulated_annealing_step,
    tabu_search_step,
    genetic_crossover,
    ant_colony_update,
    particle_swarm_update,
    iterated_local_search_step,
)

__all__ = [
    # Registry
    "OperatorRegistry",
    "ExecutionContext",
    "OperatorResult",
    "OperatorCategory",
    "create_context",
    # Construction (10)
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
    # Local Search (8)
    "two_opt",
    "three_opt",
    "or_opt",
    "swap_operator",
    "insert_operator",
    "invert_operator",
    "lin_kernighan",
    "variable_neighborhood_descent",
    # Perturbation (6)
    "double_bridge",
    "random_segment_shuffle",
    "guided_mutation",
    "ruin_recreate",
    "large_neighborhood_search",
    "adaptive_mutation",
    # Meta-heuristic (6)
    "simulated_annealing_step",
    "tabu_search_step",
    "genetic_crossover",
    "ant_colony_update",
    "particle_swarm_update",
    "iterated_local_search_step",
]
