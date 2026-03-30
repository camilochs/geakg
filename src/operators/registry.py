"""Operator Registry - Connects AKG nodes with real implementations.

This module provides a central registry that:
1. Maps operator IDs to their implementations
2. Computes adaptive parameters based on problem context
3. Manages operator state (for meta-heuristics)
4. Provides a clean execution interface
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias

from src.operators.base import Tour, DistanceMatrix, Coordinates, calculate_tour_cost

# Import all operators
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
    SimulatedAnnealingState,
    TabuList,
    AntColonyState,
    ParticleSwarmState,
)


class OperatorCategory(str, Enum):
    """Categories of operators."""
    CONSTRUCTION = "construction"
    LOCAL_SEARCH = "local_search"
    PERTURBATION = "perturbation"
    META_HEURISTIC = "meta_heuristic"


@dataclass
class ExecutionContext:
    """Context for operator execution.

    Contains all information needed to compute adaptive parameters.
    """
    # Problem info
    problem_size: int
    distance_matrix: DistanceMatrix
    coordinates: Coordinates | None = None

    # Current state
    current_tour: Tour | None = None
    current_cost: float | None = None

    # Search state
    iteration: int = 0
    generation: int = 0
    evaluations: int = 0

    # Best found
    best_tour: Tour | None = None
    best_cost: float | None = None

    # For genetic operators
    population: list[Tour] | None = None

    # Meta-heuristic states (persisted across calls)
    sa_state: SimulatedAnnealingState | None = None
    tabu_list: TabuList | None = None
    aco_state: AntColonyState | None = None
    pso_state: ParticleSwarmState | None = None

    # History for adaptive operators
    success_history: list[bool] = field(default_factory=list)
    edge_frequencies: dict[tuple[int, int], float] | None = None

    def update_cost(self):
        """Update current_cost from current_tour if needed."""
        if self.current_tour and self.current_cost is None:
            self.current_cost = calculate_tour_cost(
                self.current_tour, self.distance_matrix
            )


@dataclass
class OperatorResult:
    """Result from operator execution."""
    tour: Tour
    cost: float
    improved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Updated states (for meta-heuristics)
    sa_state: SimulatedAnnealingState | None = None
    tabu_list: TabuList | None = None
    aco_state: AntColonyState | None = None
    pso_state: ParticleSwarmState | None = None


# Type for parameter computation functions
ParamComputer: TypeAlias = Callable[[ExecutionContext], dict[str, Any]]


@dataclass
class OperatorEntry:
    """Registry entry for an operator."""
    id: str
    function: Callable
    category: OperatorCategory
    compute_params: ParamComputer
    description: str = ""
    requires_tour: bool = True  # False for construction operators


class OperatorRegistry:
    """Central registry connecting AKG operator IDs to implementations.

    Usage:
        registry = OperatorRegistry()

        context = ExecutionContext(
            problem_size=50,
            distance_matrix=dm,
            current_tour=tour,
        )

        result = registry.execute("two_opt", context)
        improved_tour = result.tour
    """

    def __init__(self):
        """Initialize registry with all operators."""
        self._operators: dict[str, OperatorEntry] = {}
        self._register_all_operators()

    def _register_all_operators(self):
        """Register all 30 operators with their parameter computers."""

        # =====================================================================
        # CONSTRUCTION OPERATORS (10)
        # =====================================================================

        self._register(
            id="greedy_nearest_neighbor",
            function=greedy_nearest_neighbor,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "start": random.randint(0, ctx.problem_size - 1),
            },
            requires_tour=False,
            description="Build tour visiting nearest unvisited city",
        )

        self._register(
            id="farthest_insertion",
            function=farthest_insertion,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
            },
            requires_tour=False,
            description="Build tour inserting farthest city",
        )

        self._register(
            id="cheapest_insertion",
            function=cheapest_insertion,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
            },
            requires_tour=False,
            description="Build tour inserting at cheapest position",
        )

        self._register(
            id="random_insertion",
            function=random_insertion,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "seed": None,  # Random each time
            },
            requires_tour=False,
            description="Build tour inserting in random order",
        )

        self._register(
            id="savings_heuristic",
            function=savings_heuristic,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "depot": 0,
                "lambda_param": 1.0,
            },
            requires_tour=False,
            description="Clarke-Wright savings algorithm",
        )

        self._register(
            id="christofides_construction",
            function=christofides_construction,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "coordinates": ctx.coordinates,
            },
            requires_tour=False,
            description="Christofides 1.5-approximation algorithm",
        )

        self._register(
            id="nearest_addition",
            function=nearest_addition,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
            },
            requires_tour=False,
            description="Build tour adding nearest city to tour",
        )

        self._register(
            id="convex_hull_start",
            function=convex_hull_start,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "coordinates": ctx.coordinates,
            },
            requires_tour=False,
            description="Start with convex hull, insert remaining",
        )

        self._register(
            id="cluster_first",
            function=cluster_first,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "coordinates": ctx.coordinates,
                # Adaptive: sqrt(n/2) clusters
                "n_clusters": max(2, int(math.sqrt(ctx.problem_size / 2))),
            },
            requires_tour=False,
            description="Cluster cities then route within clusters",
        )

        self._register(
            id="sweep_algorithm",
            function=sweep_algorithm,
            category=OperatorCategory.CONSTRUCTION,
            compute_params=lambda ctx: {
                "distance_matrix": ctx.distance_matrix,
                "coordinates": ctx.coordinates,
                "depot": 0,
                "start_angle": random.uniform(0, 2 * math.pi),
            },
            requires_tour=False,
            description="Order cities by angle from depot",
        )

        # =====================================================================
        # LOCAL SEARCH OPERATORS (8)
        # =====================================================================

        self._register(
            id="two_opt",
            function=two_opt,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=self._compute_two_opt_params,
            description="2-opt edge exchange",
        )

        self._register(
            id="three_opt",
            function=three_opt,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=self._compute_three_opt_params,
            description="3-opt edge exchange",
        )

        self._register(
            id="or_opt",
            function=or_opt,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "segment_sizes": [1, 2, 3],
                # Adaptive iterations based on problem size
                "max_iterations": min(500, ctx.problem_size * 5),
            },
            description="Relocate segments of 1-3 cities",
        )

        self._register(
            id="swap",
            function=swap_operator,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                # Adaptive iterations
                "max_iterations": min(500, ctx.problem_size * 5),
            },
            description="Swap positions of two cities",
        )

        self._register(
            id="insert",
            function=insert_operator,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "max_iterations": min(500, ctx.problem_size * 5),
            },
            description="Remove and reinsert city at best position",
        )

        self._register(
            id="invert",
            function=invert_operator,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "max_iterations": min(500, ctx.problem_size * 5),
            },
            description="Reverse segment of tour",
        )

        self._register(
            id="lin_kernighan",
            function=lin_kernighan,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=self._compute_lk_params,
            description="Variable-depth edge exchange",
        )

        self._register(
            id="variable_neighborhood",
            function=variable_neighborhood_descent,
            category=OperatorCategory.LOCAL_SEARCH,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "neighborhoods": ["swap", "insert", "two_opt"],
                "max_iterations": min(100, ctx.problem_size),
            },
            description="Systematically explore multiple neighborhoods",
        )

        # =====================================================================
        # PERTURBATION OPERATORS (6)
        # =====================================================================

        self._register(
            id="double_bridge",
            function=double_bridge,
            category=OperatorCategory.PERTURBATION,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
            },
            description="4-edge reconnection (ILS classic)",
        )

        self._register(
            id="random_segment_shuffle",
            function=random_segment_shuffle,
            category=OperatorCategory.PERTURBATION,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                # Adaptive segments: more for larger problems
                "n_segments": max(3, min(8, ctx.problem_size // 10)),
            },
            description="Shuffle tour segments",
        )

        self._register(
            id="guided_mutation",
            function=guided_mutation,
            category=OperatorCategory.PERTURBATION,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "edge_frequencies": ctx.edge_frequencies,
                # Adaptive: more mutation early, less later
                "mutation_strength": max(0.1, 0.4 - ctx.generation * 0.01),
            },
            description="Mutate based on edge frequency",
        )

        self._register(
            id="ruin_recreate",
            function=ruin_recreate,
            category=OperatorCategory.PERTURBATION,
            compute_params=self._compute_ruin_recreate_params,
            description="Remove portion and rebuild",
        )

        self._register(
            id="large_neighborhood_search",
            function=large_neighborhood_search,
            category=OperatorCategory.PERTURBATION,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                # Adaptive: larger destruction for diversification
                "destroy_fraction": 0.2 + random.uniform(0, 0.2),
            },
            description="Destroy and repair with multiple strategies",
        )

        self._register(
            id="adaptive_mutation",
            function=adaptive_mutation,
            category=OperatorCategory.PERTURBATION,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "initial_rate": 0.1,
                "success_history": ctx.success_history,
            },
            description="Self-adjusting mutation strength",
        )

        # =====================================================================
        # META-HEURISTIC OPERATORS (6)
        # =====================================================================

        self._register(
            id="simulated_annealing_step",
            function=self._execute_sa_step,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=self._compute_sa_params,
            description="SA step with temperature schedule",
        )

        self._register(
            id="tabu_search_step",
            function=self._execute_tabu_step,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=self._compute_tabu_params,
            description="Tabu search step with aspiration",
        )

        self._register(
            id="genetic_crossover",
            function=self._execute_crossover,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=lambda ctx: {
                "tour1": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "crossover_type": random.choice(["order", "pmx", "cycle"]),
            },
            description="Genetic crossover (OX/PMX/CX)",
        )

        self._register(
            id="ant_colony_update",
            function=self._execute_aco_step,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=self._compute_aco_params,
            description="ACO pheromone update",
        )

        self._register(
            id="particle_swarm_update",
            function=self._execute_pso_step,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=self._compute_pso_params,
            description="PSO position update",
        )

        self._register(
            id="iterated_local_search",
            function=self._execute_ils_step,
            category=OperatorCategory.META_HEURISTIC,
            compute_params=lambda ctx: {
                "tour": ctx.current_tour,
                "distance_matrix": ctx.distance_matrix,
                "perturbation_op": "double_bridge",
                "local_search_op": "two_opt",
                "acceptance": "improving",
                "best_tour": ctx.best_tour,
                "best_cost": ctx.best_cost,
            },
            description="ILS: perturb + local search",
        )

    def _register(
        self,
        id: str,
        function: Callable,
        category: OperatorCategory,
        compute_params: ParamComputer,
        requires_tour: bool = True,
        description: str = "",
    ):
        """Register an operator."""
        self._operators[id] = OperatorEntry(
            id=id,
            function=function,
            category=category,
            compute_params=compute_params,
            description=description,
            requires_tour=requires_tour,
        )

    # =========================================================================
    # ADAPTIVE PARAMETER COMPUTERS
    # =========================================================================

    def _compute_two_opt_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for 2-opt."""
        n = ctx.problem_size

        # More iterations for small problems, cap for large
        if n < 50:
            max_iter = n * 20
        elif n < 200:
            max_iter = n * 10
        else:
            max_iter = min(2000, n * 5)

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "max_iterations": max_iter,
            "first_improvement": n > 100,  # First improvement for large instances
        }

    def _compute_three_opt_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for 3-opt."""
        n = ctx.problem_size

        # 3-opt is O(n³), so limit iterations for large problems
        if n < 30:
            max_iter = 500
        elif n < 100:
            max_iter = 200
        else:
            max_iter = 50  # Very limited for large instances

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "max_iterations": max_iter,
        }

    def _compute_lk_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for Lin-Kernighan."""
        n = ctx.problem_size

        # Depth and iterations based on problem size
        if n < 50:
            max_depth = 5
            max_iter = 100
        elif n < 200:
            max_depth = 4
            max_iter = 50
        else:
            max_depth = 3
            max_iter = 20

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "max_depth": max_depth,
            "max_iterations": max_iter,
            "backtracking": n < 100,
        }

    def _compute_ruin_recreate_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for ruin and recreate."""
        # Adaptive ruin fraction: more aggressive if stuck
        base_fraction = 0.3

        if len(ctx.success_history) >= 5:
            recent_success = sum(ctx.success_history[-5:]) / 5
            if recent_success < 0.2:
                # Stuck: destroy more
                base_fraction = 0.4 + random.uniform(0, 0.1)
            elif recent_success > 0.6:
                # Doing well: destroy less
                base_fraction = 0.2 + random.uniform(0, 0.1)

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "ruin_fraction": base_fraction,
            "recreate_method": "greedy",
        }

    def _compute_sa_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for Simulated Annealing."""
        ctx.update_cost()

        # Initial temperature relative to solution cost
        if ctx.current_cost and ctx.current_cost > 0:
            initial_temp = ctx.current_cost * 0.05  # 5% of cost
        else:
            initial_temp = 100.0

        # Cooling rate: slower for larger problems
        n = ctx.problem_size
        if n < 50:
            cooling_rate = 0.99
        elif n < 200:
            cooling_rate = 0.995
        else:
            cooling_rate = 0.998

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "sa_state": ctx.sa_state,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "neighbor_operator": "two_opt_move",
        }

    def _compute_tabu_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for Tabu Search."""
        n = ctx.problem_size

        # Tenure proportional to sqrt(n)
        tenure = max(5, int(math.sqrt(n) * 1.5))

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "tabu_list": ctx.tabu_list,
            "tabu_tenure": tenure,
            "aspiration": True,
            "best_cost": ctx.best_cost,
        }

    def _compute_aco_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for Ant Colony."""
        n = ctx.problem_size

        # Evaporation: higher for small, lower for large
        if n < 50:
            evap = 0.2
        elif n < 200:
            evap = 0.1
        else:
            evap = 0.05

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "aco_state": ctx.aco_state,
            "evaporation_rate": evap,
            "q": 1.0,
        }

    def _compute_pso_params(self, ctx: ExecutionContext) -> dict:
        """Compute adaptive parameters for Particle Swarm."""
        # Inertia decreases over generations (more exploitation later)
        base_inertia = 0.9 - min(0.4, ctx.generation * 0.01)

        return {
            "tour": ctx.current_tour,
            "distance_matrix": ctx.distance_matrix,
            "pso_state": ctx.pso_state,
            "inertia": base_inertia,
            "cognitive": 1.5,
            "social": 1.5,
        }

    # =========================================================================
    # META-HEURISTIC EXECUTION WRAPPERS
    # =========================================================================

    def _execute_sa_step(self, **kwargs) -> tuple[Tour, SimulatedAnnealingState]:
        """Execute SA step and return results."""
        return simulated_annealing_step(**kwargs)

    def _execute_tabu_step(self, **kwargs) -> tuple[Tour, TabuList, float]:
        """Execute Tabu step and return results."""
        return tabu_search_step(**kwargs)

    def _execute_crossover(self, **kwargs) -> Tour:
        """Execute genetic crossover."""
        tour1 = kwargs.pop("tour1")
        dm = kwargs["distance_matrix"]

        # Select second parent from population or generate one
        population = kwargs.pop("population", None)
        if population and len(population) > 1:
            tour2 = random.choice([t for t in population if t != tour1])
        else:
            # Generate a different tour
            tour2 = random_insertion(dm)

        return genetic_crossover(tour1, tour2, **kwargs)

    def _execute_aco_step(self, **kwargs) -> tuple[Tour, AntColonyState]:
        """Execute ACO step and return results."""
        return ant_colony_update(**kwargs)

    def _execute_pso_step(self, **kwargs) -> tuple[Tour, ParticleSwarmState]:
        """Execute PSO step and return results."""
        return particle_swarm_update(**kwargs)

    def _execute_ils_step(self, **kwargs) -> tuple[Tour, Tour | None, float | None]:
        """Execute ILS step and return results."""
        return iterated_local_search_step(**kwargs)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def execute(
        self,
        operator_id: str,
        context: ExecutionContext,
    ) -> OperatorResult:
        """Execute an operator with adaptive parameters.

        Args:
            operator_id: ID of the operator (must match AKG node ID)
            context: Execution context with problem info and state

        Returns:
            OperatorResult with the new tour and updated state
        """
        if operator_id not in self._operators:
            raise ValueError(f"Unknown operator: {operator_id}")

        entry = self._operators[operator_id]

        # Validate that we have a tour for non-construction operators
        if entry.requires_tour and context.current_tour is None:
            raise ValueError(
                f"Operator '{operator_id}' requires a tour but none provided"
            )

        # Compute adaptive parameters
        params = entry.compute_params(context)

        # Execute operator
        result = entry.function(**params)

        # Process result based on operator type
        return self._process_result(entry, result, context)

    def _process_result(
        self,
        entry: OperatorEntry,
        result: Any,
        context: ExecutionContext,
    ) -> OperatorResult:
        """Process operator result into standardized OperatorResult."""

        # Handle different return types
        if isinstance(result, tuple):
            # Meta-heuristics return tuples with state
            if entry.id == "simulated_annealing_step":
                tour, sa_state = result
                cost = calculate_tour_cost(tour, context.distance_matrix)
                return OperatorResult(
                    tour=tour,
                    cost=cost,
                    improved=cost < (context.current_cost or float("inf")),
                    sa_state=sa_state,
                )
            elif entry.id == "tabu_search_step":
                tour, tabu_list, best_cost = result
                cost = calculate_tour_cost(tour, context.distance_matrix)
                return OperatorResult(
                    tour=tour,
                    cost=cost,
                    improved=cost < (context.current_cost or float("inf")),
                    tabu_list=tabu_list,
                    metadata={"best_cost": best_cost},
                )
            elif entry.id == "ant_colony_update":
                tour, aco_state = result
                cost = calculate_tour_cost(tour, context.distance_matrix)
                return OperatorResult(
                    tour=tour,
                    cost=cost,
                    improved=cost < (context.current_cost or float("inf")),
                    aco_state=aco_state,
                )
            elif entry.id == "particle_swarm_update":
                tour, pso_state = result
                cost = calculate_tour_cost(tour, context.distance_matrix)
                return OperatorResult(
                    tour=tour,
                    cost=cost,
                    improved=cost < (context.current_cost or float("inf")),
                    pso_state=pso_state,
                )
            elif entry.id == "iterated_local_search":
                tour, best_tour, best_cost = result
                cost = calculate_tour_cost(tour, context.distance_matrix)
                return OperatorResult(
                    tour=tour,
                    cost=cost,
                    improved=cost < (context.current_cost or float("inf")),
                    metadata={"best_tour": best_tour, "best_cost": best_cost},
                )

        # Simple operators return just a tour
        tour = result
        cost = calculate_tour_cost(tour, context.distance_matrix)

        improved = False
        if context.current_cost is not None:
            improved = cost < context.current_cost - 1e-10

        return OperatorResult(
            tour=tour,
            cost=cost,
            improved=improved,
        )

    def get_operator(self, operator_id: str) -> OperatorEntry | None:
        """Get operator entry by ID."""
        return self._operators.get(operator_id)

    def get_operators_by_category(
        self,
        category: OperatorCategory
    ) -> list[OperatorEntry]:
        """Get all operators of a category."""
        return [
            entry for entry in self._operators.values()
            if entry.category == category
        ]

    def list_operators(self) -> list[str]:
        """List all registered operator IDs."""
        return list(self._operators.keys())

    def __contains__(self, operator_id: str) -> bool:
        """Check if operator is registered."""
        return operator_id in self._operators

    def __len__(self) -> int:
        """Number of registered operators."""
        return len(self._operators)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_context(
    distance_matrix: DistanceMatrix,
    coordinates: Coordinates | None = None,
    tour: Tour | None = None,
) -> ExecutionContext:
    """Create an execution context from basic inputs.

    Args:
        distance_matrix: Distance matrix
        coordinates: Optional coordinates
        tour: Optional current tour

    Returns:
        ExecutionContext ready for use
    """
    n = len(distance_matrix)

    ctx = ExecutionContext(
        problem_size=n,
        distance_matrix=distance_matrix,
        coordinates=coordinates,
        current_tour=tour,
    )

    if tour:
        ctx.current_cost = calculate_tour_cost(tour, distance_matrix)

    return ctx
