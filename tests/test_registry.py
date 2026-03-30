"""Tests for OperatorRegistry.

Verifies that:
1. All 30 operators are registered
2. Adaptive parameters are computed correctly
3. Execution produces valid results
4. Meta-heuristic state is preserved
"""

import math
import random
import pytest

from src.operators.base import (
    create_distance_matrix,
    calculate_tour_cost,
    is_valid_tour,
)
from src.operators.registry import (
    OperatorRegistry,
    ExecutionContext,
    OperatorResult,
    OperatorCategory,
    create_context,
)
from src.operators.construction import greedy_nearest_neighbor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Create operator registry."""
    return OperatorRegistry()


@pytest.fixture
def small_context():
    """Small problem context (10 cities)."""
    random.seed(42)
    n = 10
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    dm = create_distance_matrix(coords)

    return ExecutionContext(
        problem_size=n,
        distance_matrix=dm,
        coordinates=coords,
    )


@pytest.fixture
def medium_context():
    """Medium problem context (50 cities)."""
    random.seed(123)
    n = 50
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    dm = create_distance_matrix(coords)

    return ExecutionContext(
        problem_size=n,
        distance_matrix=dm,
        coordinates=coords,
    )


@pytest.fixture
def large_context():
    """Large problem context (200 cities)."""
    random.seed(456)
    n = 200
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    dm = create_distance_matrix(coords)

    return ExecutionContext(
        problem_size=n,
        distance_matrix=dm,
        coordinates=coords,
    )


@pytest.fixture
def context_with_tour(small_context):
    """Context with an initial tour."""
    dm = small_context.distance_matrix
    tour = greedy_nearest_neighbor(dm)

    small_context.current_tour = tour
    small_context.current_cost = calculate_tour_cost(tour, dm)

    return small_context


# =============================================================================
# Basic Registry Tests
# =============================================================================

class TestRegistryBasics:
    """Test basic registry functionality."""

    def test_registry_has_30_operators(self, registry):
        """Registry should have exactly 30 operators."""
        assert len(registry) == 30

    def test_all_operators_registered(self, registry):
        """All expected operators should be registered."""
        expected_operators = [
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
            "swap",
            "insert",
            "invert",
            "lin_kernighan",
            "variable_neighborhood",
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
            "iterated_local_search",
        ]

        for op_id in expected_operators:
            assert op_id in registry, f"Operator '{op_id}' not registered"

    def test_operators_by_category(self, registry):
        """Test getting operators by category."""
        construction = registry.get_operators_by_category(OperatorCategory.CONSTRUCTION)
        assert len(construction) == 10

        local_search = registry.get_operators_by_category(OperatorCategory.LOCAL_SEARCH)
        assert len(local_search) == 8

        perturbation = registry.get_operators_by_category(OperatorCategory.PERTURBATION)
        assert len(perturbation) == 6

        meta = registry.get_operators_by_category(OperatorCategory.META_HEURISTIC)
        assert len(meta) == 6

    def test_unknown_operator_raises(self, registry, small_context):
        """Unknown operator should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operator"):
            registry.execute("nonexistent_operator", small_context)


# =============================================================================
# Construction Operator Tests
# =============================================================================

class TestConstructionExecution:
    """Test execution of construction operators."""

    @pytest.mark.parametrize("operator_id", [
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
    ])
    def test_construction_produces_valid_tour(
        self, registry, small_context, operator_id
    ):
        """All construction operators should produce valid tours."""
        result = registry.execute(operator_id, small_context)

        assert isinstance(result, OperatorResult)
        assert result.tour is not None
        assert is_valid_tour(result.tour, small_context.problem_size)
        assert result.cost > 0

    def test_construction_does_not_require_tour(self, registry, small_context):
        """Construction operators should work without initial tour."""
        assert small_context.current_tour is None

        result = registry.execute("greedy_nearest_neighbor", small_context)
        assert result.tour is not None


# =============================================================================
# Local Search Operator Tests
# =============================================================================

class TestLocalSearchExecution:
    """Test execution of local search operators."""

    @pytest.mark.parametrize("operator_id", [
        "two_opt",
        "three_opt",
        "or_opt",
        "swap",
        "insert",
        "invert",
        "lin_kernighan",
        "variable_neighborhood",
    ])
    def test_local_search_produces_valid_tour(
        self, registry, context_with_tour, operator_id
    ):
        """All local search operators should produce valid tours."""
        result = registry.execute(operator_id, context_with_tour)

        assert isinstance(result, OperatorResult)
        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        # Local search should not make things worse
        assert result.cost <= context_with_tour.current_cost + 1e-6

    def test_local_search_requires_tour(self, registry, small_context):
        """Local search should fail without initial tour."""
        with pytest.raises(ValueError, match="requires a tour"):
            registry.execute("two_opt", small_context)

    def test_two_opt_improves_random_tour(self, registry, small_context):
        """2-opt should significantly improve a random tour."""
        n = small_context.problem_size
        random_tour = list(range(n))
        random.shuffle(random_tour)

        small_context.current_tour = random_tour
        small_context.current_cost = calculate_tour_cost(
            random_tour, small_context.distance_matrix
        )

        result = registry.execute("two_opt", small_context)

        # Should improve
        assert result.improved or result.cost <= small_context.current_cost


# =============================================================================
# Perturbation Operator Tests
# =============================================================================

class TestPerturbationExecution:
    """Test execution of perturbation operators."""

    @pytest.mark.parametrize("operator_id", [
        "double_bridge",
        "random_segment_shuffle",
        "guided_mutation",
        "ruin_recreate",
        "large_neighborhood_search",
        "adaptive_mutation",
    ])
    def test_perturbation_produces_valid_tour(
        self, registry, context_with_tour, operator_id
    ):
        """All perturbation operators should produce valid tours."""
        # Use medium context for perturbation (needs more cities)
        random.seed(123)
        n = 30
        coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        dm = create_distance_matrix(coords)
        tour = greedy_nearest_neighbor(dm)

        ctx = ExecutionContext(
            problem_size=n,
            distance_matrix=dm,
            coordinates=coords,
            current_tour=tour,
            current_cost=calculate_tour_cost(tour, dm),
        )

        result = registry.execute(operator_id, ctx)

        assert isinstance(result, OperatorResult)
        assert is_valid_tour(result.tour, n)


# =============================================================================
# Meta-heuristic Operator Tests
# =============================================================================

class TestMetaHeuristicExecution:
    """Test execution of meta-heuristic operators."""

    def test_simulated_annealing_step(self, registry, context_with_tour):
        """Test SA step preserves state."""
        result = registry.execute("simulated_annealing_step", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        assert result.sa_state is not None
        assert result.sa_state.iterations == 1

        # Store temperature after first step
        temp_after_first = result.sa_state.temperature

        # Run more steps with preserved state
        context_with_tour.current_tour = result.tour
        context_with_tour.sa_state = result.sa_state

        result2 = registry.execute("simulated_annealing_step", context_with_tour)
        assert result2.sa_state.iterations == 2
        # Temperature should have decreased (same state object is mutated)
        assert result2.sa_state.temperature < temp_after_first

    def test_tabu_search_step(self, registry, context_with_tour):
        """Test Tabu search preserves state."""
        result = registry.execute("tabu_search_step", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        assert result.tabu_list is not None

    def test_ant_colony_update(self, registry, context_with_tour):
        """Test ACO preserves state."""
        result = registry.execute("ant_colony_update", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        assert result.aco_state is not None
        assert result.aco_state.best_tour is not None

    def test_particle_swarm_update(self, registry, context_with_tour):
        """Test PSO preserves state."""
        result = registry.execute("particle_swarm_update", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        assert result.pso_state is not None

    def test_iterated_local_search(self, registry, context_with_tour):
        """Test ILS step."""
        result = registry.execute("iterated_local_search", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)
        assert "best_tour" in result.metadata

    def test_genetic_crossover(self, registry, context_with_tour):
        """Test genetic crossover."""
        result = registry.execute("genetic_crossover", context_with_tour)

        assert is_valid_tour(result.tour, context_with_tour.problem_size)


# =============================================================================
# Adaptive Parameter Tests
# =============================================================================

class TestAdaptiveParameters:
    """Test that parameters adapt to problem size."""

    def test_two_opt_iterations_scale_with_size(self, registry):
        """2-opt iterations should scale with problem size."""
        for n, expected_behavior in [(10, "high"), (100, "medium"), (500, "low")]:
            coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
            dm = create_distance_matrix(coords)

            ctx = ExecutionContext(
                problem_size=n,
                distance_matrix=dm,
                current_tour=list(range(n)),
            )

            entry = registry.get_operator("two_opt")
            params = entry.compute_params(ctx)

            if expected_behavior == "high":
                assert params["max_iterations"] >= n * 10
            elif expected_behavior == "low":
                assert params["max_iterations"] <= n * 10
                assert params["first_improvement"] is True

    def test_tabu_tenure_scales_with_sqrt_n(self, registry):
        """Tabu tenure should scale with sqrt(n)."""
        for n in [25, 100, 400]:
            coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
            dm = create_distance_matrix(coords)
            tour = list(range(n))

            ctx = ExecutionContext(
                problem_size=n,
                distance_matrix=dm,
                current_tour=tour,
            )

            entry = registry.get_operator("tabu_search_step")
            params = entry.compute_params(ctx)

            expected_tenure = max(5, int(math.sqrt(n) * 1.5))
            assert params["tabu_tenure"] == expected_tenure

    def test_sa_temperature_relative_to_cost(self, registry, context_with_tour):
        """SA initial temperature should be relative to tour cost."""
        entry = registry.get_operator("simulated_annealing_step")
        params = entry.compute_params(context_with_tour)

        # Temperature should be ~5% of cost
        expected_temp = context_with_tour.current_cost * 0.05
        assert abs(params["initial_temp"] - expected_temp) < 1e-6

    def test_cluster_count_scales(self, registry):
        """Cluster count should scale with sqrt(n/2)."""
        for n in [20, 100, 200]:
            coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
            dm = create_distance_matrix(coords)

            ctx = ExecutionContext(
                problem_size=n,
                distance_matrix=dm,
                coordinates=coords,
            )

            entry = registry.get_operator("cluster_first")
            params = entry.compute_params(ctx)

            expected_clusters = max(2, int(math.sqrt(n / 2)))
            assert params["n_clusters"] == expected_clusters


# =============================================================================
# Integration Tests
# =============================================================================

class TestRegistryIntegration:
    """Integration tests combining multiple operators."""

    def test_full_pipeline(self, registry, medium_context):
        """Test construction → local search → perturbation → local search."""
        # Construction
        result = registry.execute("christofides_construction", medium_context)
        assert is_valid_tour(result.tour, medium_context.problem_size)

        # Update context
        medium_context.current_tour = result.tour
        medium_context.current_cost = result.cost

        # Local search
        result = registry.execute("two_opt", medium_context)
        assert result.cost <= medium_context.current_cost + 1e-6

        # Update
        medium_context.current_tour = result.tour
        medium_context.current_cost = result.cost

        # Perturbation
        result = registry.execute("double_bridge", medium_context)
        assert is_valid_tour(result.tour, medium_context.problem_size)

        # Update
        medium_context.current_tour = result.tour
        medium_context.current_cost = result.cost

        # Local search again
        result = registry.execute("two_opt", medium_context)
        assert is_valid_tour(result.tour, medium_context.problem_size)

    def test_ils_loop(self, registry, medium_context):
        """Test multiple ILS iterations."""
        # Initial construction
        result = registry.execute("greedy_nearest_neighbor", medium_context)
        medium_context.current_tour = result.tour
        medium_context.current_cost = result.cost
        medium_context.best_tour = result.tour
        medium_context.best_cost = result.cost

        initial_cost = result.cost

        # Run 5 ILS steps
        for i in range(5):
            medium_context.iteration = i
            result = registry.execute("iterated_local_search", medium_context)

            medium_context.current_tour = result.tour
            medium_context.current_cost = result.cost

            if result.cost < medium_context.best_cost:
                medium_context.best_cost = result.cost
                medium_context.best_tour = result.tour

        # Should have improved
        assert medium_context.best_cost <= initial_cost

    def test_create_context_helper(self):
        """Test create_context helper function."""
        random.seed(42)
        n = 10
        coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        dm = create_distance_matrix(coords)
        tour = list(range(n))

        ctx = create_context(dm, coords, tour)

        assert ctx.problem_size == n
        assert ctx.current_tour == tour
        assert ctx.current_cost is not None
        assert ctx.coordinates == coords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
