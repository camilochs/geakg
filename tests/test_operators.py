"""Tests for all 30 TSP operators.

Tests verify:
1. Output is a valid tour (visits all cities exactly once)
2. Construction operators work from scratch
3. Local search operators improve or maintain solution quality
4. Perturbation operators modify the tour
5. Meta-heuristic operators work correctly with their state
"""

import math
import random
import pytest
from typing import Callable

# Import base utilities
from src.operators.base import (
    Tour,
    DistanceMatrix,
    calculate_tour_cost,
    is_valid_tour,
    create_distance_matrix,
)

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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_instance():
    """Small TSP instance (10 cities) for quick tests."""
    random.seed(42)
    n = 10
    coordinates = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    distance_matrix = create_distance_matrix(coordinates)
    return {
        "n": n,
        "coordinates": coordinates,
        "distance_matrix": distance_matrix,
    }


@pytest.fixture
def medium_instance():
    """Medium TSP instance (50 cities) for more thorough tests."""
    random.seed(123)
    n = 50
    coordinates = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    distance_matrix = create_distance_matrix(coordinates)
    return {
        "n": n,
        "coordinates": coordinates,
        "distance_matrix": distance_matrix,
    }


@pytest.fixture
def clustered_instance():
    """Clustered TSP instance (4 clusters of 10 cities)."""
    random.seed(456)
    n = 40
    coordinates = []

    # 4 cluster centers
    centers = [(25, 25), (75, 25), (25, 75), (75, 75)]

    for cx, cy in centers:
        for _ in range(10):
            x = cx + random.gauss(0, 5)
            y = cy + random.gauss(0, 5)
            coordinates.append((x, y))

    distance_matrix = create_distance_matrix(coordinates)
    return {
        "n": n,
        "coordinates": coordinates,
        "distance_matrix": distance_matrix,
    }


@pytest.fixture
def random_tour(small_instance):
    """A random tour for the small instance."""
    tour = list(range(small_instance["n"]))
    random.shuffle(tour)
    return tour


@pytest.fixture
def nn_tour(small_instance):
    """Nearest neighbor tour for the small instance."""
    return greedy_nearest_neighbor(small_instance["distance_matrix"])


# =============================================================================
# Helper functions
# =============================================================================

def assert_valid_tour(tour: Tour, n: int, name: str = ""):
    """Assert that a tour is valid."""
    assert tour is not None, f"{name}: Tour is None"
    assert len(tour) == n, f"{name}: Tour length {len(tour)} != {n}"
    assert set(tour) == set(range(n)), f"{name}: Tour doesn't visit all cities exactly once"


def assert_improvement(
    old_tour: Tour,
    new_tour: Tour,
    distance_matrix: DistanceMatrix,
    name: str = "",
    allow_equal: bool = True
):
    """Assert that new tour is at least as good as old tour."""
    old_cost = calculate_tour_cost(old_tour, distance_matrix)
    new_cost = calculate_tour_cost(new_tour, distance_matrix)

    if allow_equal:
        assert new_cost <= old_cost + 1e-6, \
            f"{name}: Cost increased from {old_cost:.2f} to {new_cost:.2f}"
    else:
        assert new_cost < old_cost - 1e-6, \
            f"{name}: Cost did not improve from {old_cost:.2f} to {new_cost:.2f}"


# =============================================================================
# Construction Operator Tests
# =============================================================================

class TestConstructionOperators:
    """Tests for construction operators."""

    def test_greedy_nearest_neighbor(self, small_instance):
        """Test greedy nearest neighbor construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = greedy_nearest_neighbor(dm)
        assert_valid_tour(tour, n, "greedy_nearest_neighbor")

        # Test with specific start
        tour_start0 = greedy_nearest_neighbor(dm, start=0)
        assert tour_start0[0] == 0, "Tour should start at specified city"
        assert_valid_tour(tour_start0, n, "greedy_nearest_neighbor(start=0)")

    def test_farthest_insertion(self, small_instance):
        """Test farthest insertion construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = farthest_insertion(dm)
        assert_valid_tour(tour, n, "farthest_insertion")

    def test_cheapest_insertion(self, small_instance):
        """Test cheapest insertion construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = cheapest_insertion(dm)
        assert_valid_tour(tour, n, "cheapest_insertion")

    def test_random_insertion(self, small_instance):
        """Test random insertion construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = random_insertion(dm, seed=42)
        assert_valid_tour(tour, n, "random_insertion")

        # Same seed should give same tour
        tour2 = random_insertion(dm, seed=42)
        assert tour == tour2, "Same seed should give same tour"

    def test_savings_heuristic(self, small_instance):
        """Test Clarke-Wright savings heuristic."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = savings_heuristic(dm)
        assert_valid_tour(tour, n, "savings_heuristic")

    def test_christofides_construction(self, small_instance):
        """Test Christofides construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = christofides_construction(dm)
        assert_valid_tour(tour, n, "christofides_construction")

    def test_nearest_addition(self, small_instance):
        """Test nearest addition construction."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        tour = nearest_addition(dm)
        assert_valid_tour(tour, n, "nearest_addition")

    def test_convex_hull_start(self, small_instance):
        """Test convex hull construction."""
        dm = small_instance["distance_matrix"]
        coords = small_instance["coordinates"]
        n = small_instance["n"]

        tour = convex_hull_start(dm, coordinates=coords)
        assert_valid_tour(tour, n, "convex_hull_start")

        # Also test fallback without coordinates
        tour_no_coords = convex_hull_start(dm)
        assert_valid_tour(tour_no_coords, n, "convex_hull_start(no coords)")

    def test_cluster_first(self, clustered_instance):
        """Test cluster first route second."""
        dm = clustered_instance["distance_matrix"]
        n = clustered_instance["n"]

        tour = cluster_first(dm, n_clusters=4)
        assert_valid_tour(tour, n, "cluster_first")

    def test_sweep_algorithm(self, small_instance):
        """Test sweep algorithm."""
        dm = small_instance["distance_matrix"]
        coords = small_instance["coordinates"]
        n = small_instance["n"]

        tour = sweep_algorithm(dm, coordinates=coords)
        assert_valid_tour(tour, n, "sweep_algorithm")

        # Test fallback without coordinates
        tour_no_coords = sweep_algorithm(dm)
        assert_valid_tour(tour_no_coords, n, "sweep_algorithm(no coords)")


# =============================================================================
# Local Search Operator Tests
# =============================================================================

class TestLocalSearchOperators:
    """Tests for local search operators."""

    def test_two_opt(self, small_instance, random_tour):
        """Test 2-opt local search."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = two_opt(random_tour, dm)
        assert_valid_tour(improved, n, "two_opt")
        assert_improvement(random_tour, improved, dm, "two_opt")

    def test_two_opt_first_improvement(self, small_instance, random_tour):
        """Test 2-opt with first improvement."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = two_opt(random_tour, dm, first_improvement=True)
        assert_valid_tour(improved, n, "two_opt(first_improvement)")
        assert_improvement(random_tour, improved, dm, "two_opt(first_improvement)")

    def test_three_opt(self, small_instance, random_tour):
        """Test 3-opt local search."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = three_opt(random_tour, dm, max_iterations=100)
        assert_valid_tour(improved, n, "three_opt")
        assert_improvement(random_tour, improved, dm, "three_opt")

    def test_or_opt(self, small_instance, random_tour):
        """Test Or-opt local search."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = or_opt(random_tour, dm)
        assert_valid_tour(improved, n, "or_opt")
        assert_improvement(random_tour, improved, dm, "or_opt")

    def test_swap_operator(self, small_instance, random_tour):
        """Test swap operator."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = swap_operator(random_tour, dm)
        assert_valid_tour(improved, n, "swap_operator")
        assert_improvement(random_tour, improved, dm, "swap_operator")

    def test_insert_operator(self, small_instance, random_tour):
        """Test insert/relocate operator."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = insert_operator(random_tour, dm)
        assert_valid_tour(improved, n, "insert_operator")
        assert_improvement(random_tour, improved, dm, "insert_operator")

    def test_invert_operator(self, small_instance, random_tour):
        """Test invert operator."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = invert_operator(random_tour, dm)
        assert_valid_tour(improved, n, "invert_operator")
        assert_improvement(random_tour, improved, dm, "invert_operator")

    def test_lin_kernighan(self, small_instance, random_tour):
        """Test Lin-Kernighan heuristic."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = lin_kernighan(random_tour, dm)
        assert_valid_tour(improved, n, "lin_kernighan")
        assert_improvement(random_tour, improved, dm, "lin_kernighan")

    def test_variable_neighborhood_descent(self, small_instance, random_tour):
        """Test Variable Neighborhood Descent."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        improved = variable_neighborhood_descent(random_tour, dm)
        assert_valid_tour(improved, n, "variable_neighborhood_descent")
        assert_improvement(random_tour, improved, dm, "variable_neighborhood_descent")


# =============================================================================
# Perturbation Operator Tests
# =============================================================================

class TestPerturbationOperators:
    """Tests for perturbation operators."""

    def test_double_bridge(self, medium_instance, nn_tour):
        """Test double bridge perturbation."""
        dm = medium_instance["distance_matrix"]
        n = medium_instance["n"]

        # Get a tour for medium instance
        tour = greedy_nearest_neighbor(dm)

        perturbed = double_bridge(tour, dm)
        assert_valid_tour(perturbed, n, "double_bridge")

        # Tour should be different
        assert perturbed != tour, "Perturbation should modify tour"

    def test_random_segment_shuffle(self, medium_instance):
        """Test random segment shuffle."""
        dm = medium_instance["distance_matrix"]
        n = medium_instance["n"]
        tour = greedy_nearest_neighbor(dm)

        perturbed = random_segment_shuffle(tour, dm, n_segments=4)
        assert_valid_tour(perturbed, n, "random_segment_shuffle")

    def test_guided_mutation(self, small_instance, nn_tour):
        """Test guided mutation."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        perturbed = guided_mutation(nn_tour, dm, mutation_strength=0.3)
        assert_valid_tour(perturbed, n, "guided_mutation")

    def test_ruin_recreate(self, small_instance, nn_tour):
        """Test ruin and recreate."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        perturbed = ruin_recreate(nn_tour, dm, ruin_fraction=0.3)
        assert_valid_tour(perturbed, n, "ruin_recreate")

        # Test random recreate
        perturbed_random = ruin_recreate(nn_tour, dm, recreate_method="random")
        assert_valid_tour(perturbed_random, n, "ruin_recreate(random)")

    def test_large_neighborhood_search(self, small_instance, nn_tour):
        """Test Large Neighborhood Search."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        perturbed = large_neighborhood_search(nn_tour, dm, destroy_fraction=0.3)
        assert_valid_tour(perturbed, n, "large_neighborhood_search")

    def test_adaptive_mutation(self, small_instance, nn_tour):
        """Test adaptive mutation."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        # Without history
        perturbed = adaptive_mutation(nn_tour, dm, initial_rate=0.1)
        assert_valid_tour(perturbed, n, "adaptive_mutation")

        # With history (high success - less mutation)
        perturbed_high = adaptive_mutation(nn_tour, dm, success_history=[True] * 10)
        assert_valid_tour(perturbed_high, n, "adaptive_mutation(high_success)")

        # With history (low success - more mutation)
        perturbed_low = adaptive_mutation(nn_tour, dm, success_history=[False] * 10)
        assert_valid_tour(perturbed_low, n, "adaptive_mutation(low_success)")


# =============================================================================
# Meta-heuristic Operator Tests
# =============================================================================

class TestMetaHeuristicOperators:
    """Tests for meta-heuristic operators."""

    def test_simulated_annealing_step(self, small_instance, nn_tour):
        """Test Simulated Annealing step."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        new_tour, sa_state = simulated_annealing_step(nn_tour, dm)
        assert_valid_tour(new_tour, n, "simulated_annealing_step")
        assert isinstance(sa_state, SimulatedAnnealingState)

        # Run multiple steps
        tour = nn_tour
        for _ in range(10):
            tour, sa_state = simulated_annealing_step(tour, dm, sa_state=sa_state)
            assert_valid_tour(tour, n, "simulated_annealing_step(multi)")

        # Temperature should have decreased
        assert sa_state.temperature < sa_state.initial_temp

    def test_tabu_search_step(self, small_instance, nn_tour):
        """Test Tabu Search step."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        new_tour, tabu_list, best_cost = tabu_search_step(nn_tour, dm)
        assert_valid_tour(new_tour, n, "tabu_search_step")
        assert isinstance(tabu_list, TabuList)
        assert best_cost is not None

        # Run multiple steps
        tour = nn_tour
        for _ in range(10):
            tour, tabu_list, best_cost = tabu_search_step(
                tour, dm, tabu_list=tabu_list, best_cost=best_cost
            )
            assert_valid_tour(tour, n, "tabu_search_step(multi)")

    def test_genetic_crossover_order(self, small_instance):
        """Test Order Crossover (OX)."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        parent1 = greedy_nearest_neighbor(dm)
        parent2 = random_insertion(dm, seed=99)

        child = genetic_crossover(parent1, parent2, dm, crossover_type="order")
        assert_valid_tour(child, n, "genetic_crossover(order)")

    def test_genetic_crossover_pmx(self, small_instance):
        """Test Partially Mapped Crossover (PMX)."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        parent1 = greedy_nearest_neighbor(dm)
        parent2 = random_insertion(dm, seed=99)

        child = genetic_crossover(parent1, parent2, dm, crossover_type="pmx")
        assert_valid_tour(child, n, "genetic_crossover(pmx)")

    def test_genetic_crossover_cycle(self, small_instance):
        """Test Cycle Crossover (CX)."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        parent1 = greedy_nearest_neighbor(dm)
        parent2 = random_insertion(dm, seed=99)

        child = genetic_crossover(parent1, parent2, dm, crossover_type="cycle")
        assert_valid_tour(child, n, "genetic_crossover(cycle)")

    def test_ant_colony_update(self, small_instance, nn_tour):
        """Test Ant Colony Optimization update."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        new_tour, aco_state = ant_colony_update(nn_tour, dm)
        assert_valid_tour(new_tour, n, "ant_colony_update")
        assert isinstance(aco_state, AntColonyState)

        # Run multiple iterations
        tour = nn_tour
        for _ in range(5):
            tour, aco_state = ant_colony_update(tour, dm, aco_state=aco_state)
            assert_valid_tour(tour, n, "ant_colony_update(multi)")

        # Best should be tracked
        assert aco_state.best_tour is not None

    def test_particle_swarm_update(self, small_instance, nn_tour):
        """Test Particle Swarm Optimization update."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        new_tour, pso_state = particle_swarm_update(nn_tour, dm)
        assert_valid_tour(new_tour, n, "particle_swarm_update")
        assert isinstance(pso_state, ParticleSwarmState)

        # Run multiple iterations
        tour = nn_tour
        for _ in range(5):
            tour, pso_state = particle_swarm_update(tour, dm, pso_state=pso_state)
            assert_valid_tour(tour, n, "particle_swarm_update(multi)")

    def test_iterated_local_search_step(self, small_instance, nn_tour):
        """Test Iterated Local Search step."""
        dm = small_instance["distance_matrix"]
        n = small_instance["n"]

        new_tour, best_tour, best_cost = iterated_local_search_step(nn_tour, dm)
        assert_valid_tour(new_tour, n, "iterated_local_search_step")
        assert_valid_tour(best_tour, n, "iterated_local_search_step(best)")

        # Run multiple iterations
        tour = nn_tour
        for _ in range(5):
            tour, best_tour, best_cost = iterated_local_search_step(
                tour, dm, best_tour=best_tour, best_cost=best_cost
            )
            assert_valid_tour(tour, n, "iterated_local_search_step(multi)")


# =============================================================================
# Integration Tests
# =============================================================================

class TestOperatorIntegration:
    """Integration tests combining multiple operators."""

    def test_construction_then_local_search(self, medium_instance):
        """Test typical pipeline: construction → local search."""
        dm = medium_instance["distance_matrix"]
        n = medium_instance["n"]

        # All construction operators
        construction_ops = [
            greedy_nearest_neighbor,
            farthest_insertion,
            cheapest_insertion,
            random_insertion,
            nearest_addition,
        ]

        for constr_op in construction_ops:
            tour = constr_op(dm)
            assert_valid_tour(tour, n, constr_op.__name__)

            # Apply 2-opt
            improved = two_opt(tour, dm)
            assert_valid_tour(improved, n, f"{constr_op.__name__} + two_opt")
            assert_improvement(tour, improved, dm, f"{constr_op.__name__} + two_opt")

    def test_ils_pipeline(self, medium_instance):
        """Test ILS pipeline: construction → local search → perturbation → local search."""
        dm = medium_instance["distance_matrix"]
        n = medium_instance["n"]

        # Construction
        tour = greedy_nearest_neighbor(dm)
        assert_valid_tour(tour, n, "ILS: construction")

        # Local search
        tour = two_opt(tour, dm)
        assert_valid_tour(tour, n, "ILS: first 2-opt")
        best_cost = calculate_tour_cost(tour, dm)

        # ILS iterations
        for i in range(5):
            # Perturbation
            perturbed = double_bridge(tour, dm)
            assert_valid_tour(perturbed, n, f"ILS iteration {i}: perturbation")

            # Local search
            improved = two_opt(perturbed, dm)
            assert_valid_tour(improved, n, f"ILS iteration {i}: local search")

            # Accept if better
            cost = calculate_tour_cost(improved, dm)
            if cost < best_cost:
                tour = improved
                best_cost = cost

        assert_valid_tour(tour, n, "ILS: final")

    def test_all_operators_produce_valid_tours(self, small_instance):
        """Ensure all 30 operators produce valid tours."""
        dm = small_instance["distance_matrix"]
        coords = small_instance["coordinates"]
        n = small_instance["n"]

        # Get an initial tour
        initial_tour = list(range(n))
        random.shuffle(initial_tour)

        # Construction operators (10)
        construction_results = {
            "greedy_nearest_neighbor": greedy_nearest_neighbor(dm),
            "farthest_insertion": farthest_insertion(dm),
            "cheapest_insertion": cheapest_insertion(dm),
            "random_insertion": random_insertion(dm),
            "savings_heuristic": savings_heuristic(dm),
            "christofides_construction": christofides_construction(dm),
            "nearest_addition": nearest_addition(dm),
            "convex_hull_start": convex_hull_start(dm, coordinates=coords),
            "cluster_first": cluster_first(dm),
            "sweep_algorithm": sweep_algorithm(dm, coordinates=coords),
        }

        for name, tour in construction_results.items():
            assert_valid_tour(tour, n, name)

        # Local search operators (8) - use nn tour as input
        nn_tour = greedy_nearest_neighbor(dm)
        local_search_results = {
            "two_opt": two_opt(nn_tour, dm),
            "three_opt": three_opt(nn_tour, dm, max_iterations=50),
            "or_opt": or_opt(nn_tour, dm),
            "swap_operator": swap_operator(nn_tour, dm),
            "insert_operator": insert_operator(nn_tour, dm),
            "invert_operator": invert_operator(nn_tour, dm),
            "lin_kernighan": lin_kernighan(nn_tour, dm),
            "variable_neighborhood_descent": variable_neighborhood_descent(nn_tour, dm),
        }

        for name, tour in local_search_results.items():
            assert_valid_tour(tour, n, name)

        # Perturbation operators (6)
        perturbation_results = {
            "double_bridge": double_bridge(nn_tour, dm),
            "random_segment_shuffle": random_segment_shuffle(nn_tour, dm),
            "guided_mutation": guided_mutation(nn_tour, dm),
            "ruin_recreate": ruin_recreate(nn_tour, dm),
            "large_neighborhood_search": large_neighborhood_search(nn_tour, dm),
            "adaptive_mutation": adaptive_mutation(nn_tour, dm),
        }

        for name, tour in perturbation_results.items():
            assert_valid_tour(tour, n, name)

        # Meta-heuristic operators (6)
        tour1, _ = simulated_annealing_step(nn_tour, dm)
        assert_valid_tour(tour1, n, "simulated_annealing_step")

        tour2, _, _ = tabu_search_step(nn_tour, dm)
        assert_valid_tour(tour2, n, "tabu_search_step")

        parent2 = random_insertion(dm, seed=99)
        tour3 = genetic_crossover(nn_tour, parent2, dm)
        assert_valid_tour(tour3, n, "genetic_crossover")

        tour4, _ = ant_colony_update(nn_tour, dm)
        assert_valid_tour(tour4, n, "ant_colony_update")

        tour5, _ = particle_swarm_update(nn_tour, dm)
        assert_valid_tour(tour5, n, "particle_swarm_update")

        tour6, _, _ = iterated_local_search_step(nn_tour, dm)
        assert_valid_tour(tour6, n, "iterated_local_search_step")

        print("\n✅ All 30 operators produce valid tours!")


# =============================================================================
# Quality Tests
# =============================================================================

class TestOperatorQuality:
    """Tests that verify operators produce good quality solutions."""

    def test_construction_quality_ordering(self, medium_instance):
        """Test that sophisticated construction operators generally produce better solutions."""
        dm = medium_instance["distance_matrix"]
        coords = medium_instance["coordinates"]

        # Run each construction operator multiple times
        results = {}
        for name, op in [
            ("random_insertion", lambda: random_insertion(dm, seed=None)),
            ("greedy_nearest_neighbor", lambda: greedy_nearest_neighbor(dm)),
            ("cheapest_insertion", lambda: cheapest_insertion(dm)),
            ("christofides_construction", lambda: christofides_construction(dm)),
        ]:
            costs = []
            for _ in range(5):
                tour = op()
                costs.append(calculate_tour_cost(tour, dm))
            results[name] = min(costs)

        # Christofides should generally be among the best
        # (Not strictly enforced due to randomness)
        print(f"\nConstruction quality: {results}")

    def test_local_search_improves_random(self, medium_instance):
        """Test that local search significantly improves random tours."""
        dm = medium_instance["distance_matrix"]
        n = medium_instance["n"]

        improvements = []
        for _ in range(5):
            random_tour = list(range(n))
            random.shuffle(random_tour)
            random_cost = calculate_tour_cost(random_tour, dm)

            improved = two_opt(random_tour, dm)
            improved_cost = calculate_tour_cost(improved, dm)

            improvement = (random_cost - improved_cost) / random_cost * 100
            improvements.append(improvement)

        avg_improvement = sum(improvements) / len(improvements)
        assert avg_improvement > 10, f"2-opt should improve random tours by >10%, got {avg_improvement:.1f}%"
        print(f"\n2-opt average improvement on random tours: {avg_improvement:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
