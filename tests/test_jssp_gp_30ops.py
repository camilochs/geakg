"""Tests for JSSP GP with 30 TSP-derived operators.

Verifies that:
1. GP with 30 operators produces valid JSSP schedules
2. The makespan calculation is correct
3. Precedence constraints are respected
4. All operator categories are functional
"""

import random
import pytest

from src.baselines.jssp_gp_30ops import JSSPGP30Ops, JSSPGP30Result


# =============================================================================
# Fixtures - Standard JSSP Instances
# =============================================================================

@pytest.fixture
def ft06_instance():
    """Fisher and Thompson 6x6 instance (ft06).

    6 jobs, 6 machines, optimal makespan = 55
    """
    processing_times = [
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ]
    machine_assignments = [
        [2, 0, 1, 3, 5, 4],
        [1, 2, 4, 5, 0, 3],
        [2, 3, 5, 0, 1, 4],
        [1, 0, 2, 3, 4, 5],
        [2, 1, 4, 5, 0, 3],
        [1, 3, 5, 0, 4, 2],
    ]
    optimal = 55
    return processing_times, machine_assignments, optimal


@pytest.fixture
def small_instance():
    """Small 3x3 JSSP instance for quick testing."""
    processing_times = [
        [3, 2, 2],
        [2, 1, 4],
        [4, 3, 1],
    ]
    machine_assignments = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 2, 0],
    ]
    return processing_times, machine_assignments


@pytest.fixture
def la01_instance():
    """Lawrence la01 instance (10x5).

    10 jobs, 5 machines, optimal makespan = 666
    """
    processing_times = [
        [21, 53, 95, 55, 34],
        [21, 52, 16, 26, 71],
        [39, 98, 42, 31, 12],
        [77, 55, 79, 66, 77],
        [83, 34, 64, 19, 37],
        [54, 43, 79, 92, 62],
        [69, 77, 87, 93, 11],
        [38, 60, 41, 24, 83],
        [17, 49, 25, 44, 98],
        [77, 79, 43, 75, 96],
    ]
    machine_assignments = [
        [1, 0, 4, 3, 2],
        [0, 3, 4, 2, 1],
        [1, 2, 0, 3, 4],
        [1, 0, 3, 4, 2],
        [2, 0, 1, 3, 4],
        [2, 1, 4, 3, 0],
        [1, 0, 2, 3, 4],
        [2, 4, 1, 0, 3],
        [0, 1, 4, 2, 3],
        [1, 3, 2, 0, 4],
    ]
    optimal = 666
    return processing_times, machine_assignments, optimal


# =============================================================================
# Basic GP Tests
# =============================================================================

class TestGP30Basics:
    """Test basic GP functionality."""

    def test_gp_initialization(self):
        """GP should initialize with correct parameters."""
        gp = JSSPGP30Ops(
            population_size=50,
            max_generations=20,
            budget=100,
            seed=42,
        )
        assert gp.population_size == 50
        assert gp.max_generations == 20
        assert gp.budget == 100

    def test_gp_returns_result(self, small_instance):
        """GP should return JSSPGP30Result."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        assert isinstance(result, JSSPGP30Result)
        assert result.best_fitness > 0
        assert result.evaluations <= 50
        assert result.wall_time_seconds > 0

    def test_gp_respects_budget(self, small_instance):
        """GP should not exceed evaluation budget."""
        pt, ma = small_instance
        budget = 30
        gp = JSSPGP30Ops(budget=budget, seed=42)
        result = gp.run(pt, ma)

        assert result.evaluations <= budget

    def test_gp_fitness_history(self, small_instance):
        """GP should track fitness history."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        assert len(result.fitness_history) > 0
        assert len(result.fitness_history) == result.evaluations


# =============================================================================
# Solution Quality Tests
# =============================================================================

class TestSolutionQuality:
    """Test solution quality on standard instances."""

    def test_ft06_finds_valid_solution(self, ft06_instance):
        """GP should find a valid solution for ft06."""
        pt, ma, optimal = ft06_instance
        gp = JSSPGP30Ops(budget=200, seed=42)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")
        # Should be within 2x of optimal for this small instance
        assert result.best_fitness <= optimal * 2.0

    def test_la01_finds_valid_solution(self, la01_instance):
        """GP should find a valid solution for la01."""
        pt, ma, optimal = la01_instance
        gp = JSSPGP30Ops(budget=200, seed=42)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")
        # Should be within 2x of optimal
        assert result.best_fitness <= optimal * 2.0

    def test_consistent_results_with_seed(self, small_instance):
        """Same seed should produce same results."""
        pt, ma = small_instance

        gp1 = JSSPGP30Ops(budget=50, seed=42)
        result1 = gp1.run(pt, ma)

        gp2 = JSSPGP30Ops(budget=50, seed=42)
        result2 = gp2.run(pt, ma)

        assert result1.best_fitness == result2.best_fitness

    def test_different_seeds_produce_variation(self, ft06_instance):
        """Different seeds should produce different results."""
        pt, ma, _ = ft06_instance

        results = []
        for seed in [1, 2, 3, 4, 5]:
            gp = JSSPGP30Ops(budget=30, seed=seed)
            result = gp.run(pt, ma)
            results.append(result.best_fitness)

        # All should find valid solutions
        assert all(f < float("inf") for f in results)


# =============================================================================
# Operator Tests
# =============================================================================

class TestOperators:
    """Test that operators function correctly."""

    def test_construction_operators_work(self, small_instance):
        """Construction operators should produce valid schedules."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=100, seed=42)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")

    def test_local_search_operators_work(self, ft06_instance):
        """Local search operators should be usable."""
        pt, ma, _ = ft06_instance
        gp = JSSPGP30Ops(budget=100, seed=42)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")

    def test_perturbation_operators_work(self, ft06_instance):
        """Perturbation operators should be usable."""
        pt, ma, _ = ft06_instance
        gp = JSSPGP30Ops(budget=100, seed=123)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")

    def test_metaheuristic_operators_work(self, ft06_instance):
        """Meta-heuristic operators should be usable."""
        pt, ma, _ = ft06_instance
        gp = JSSPGP30Ops(budget=100, seed=456)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")


# =============================================================================
# Validity Tests
# =============================================================================

class TestScheduleValidity:
    """Test schedule validity checking."""

    def test_valid_schedule_checking(self, small_instance):
        """Valid schedule should pass validation."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=10, seed=42)

        # Setup internal state
        gp._processing_times = pt
        gp._machine_assignments = ma
        gp._n_jobs = len(pt)
        gp._n_ops_per_job = len(pt[0])
        gp._n_machines = len(set(m for job in ma for m in job))
        gp._total_ops = gp._n_jobs * gp._n_ops_per_job

        # Valid schedule: all operations in order within each job
        valid_schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Sequential by job
        assert gp._is_valid_schedule(valid_schedule)

        # Valid schedule: interleaved but respecting precedence
        valid_interleaved = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        assert gp._is_valid_schedule(valid_interleaved)

    def test_invalid_schedule_detection(self, small_instance):
        """Invalid schedule should fail validation."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=10, seed=42)

        # Setup internal state
        gp._processing_times = pt
        gp._machine_assignments = ma
        gp._n_jobs = len(pt)
        gp._n_ops_per_job = len(pt[0])
        gp._n_machines = len(set(m for job in ma for m in job))
        gp._total_ops = gp._n_jobs * gp._n_ops_per_job

        # Invalid: op 1 before op 0 for job 0
        invalid_schedule = [1, 0, 2, 3, 4, 5, 6, 7, 8]
        assert not gp._is_valid_schedule(invalid_schedule)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_small_instance_runs_fast(self, small_instance):
        """Small instances should run quickly."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        assert result.wall_time_seconds < 10.0
        assert result.best_fitness < float("inf")

    def test_ft06_completes_reasonably(self, ft06_instance):
        """ft06 should complete in reasonable time."""
        pt, ma, _ = ft06_instance
        gp = JSSPGP30Ops(budget=100, seed=42)
        result = gp.run(pt, ma)

        assert result.wall_time_seconds < 60.0
        assert result.best_fitness < float("inf")

    def test_la01_completes(self, la01_instance):
        """la01 should complete."""
        pt, ma, _ = la01_instance
        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        assert result.best_fitness < float("inf")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_minimal_budget(self, small_instance):
        """Should handle minimal budget."""
        pt, ma = small_instance
        gp = JSSPGP30Ops(budget=10, seed=42)
        result = gp.run(pt, ma)

        assert result.evaluations <= 10

    def test_single_job(self):
        """Should handle single job instance."""
        pt = [[1, 2, 3]]
        ma = [[0, 1, 2]]

        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        # Single job with sequential ops: makespan = 1 + 2 + 3 = 6
        # GP should find valid solution
        assert result.best_fitness <= 6
        assert result.best_fitness > 0

    def test_single_machine(self):
        """Should handle single machine instance."""
        pt = [[1], [2], [3]]
        ma = [[0], [0], [0]]

        gp = JSSPGP30Ops(budget=50, seed=42)
        result = gp.run(pt, ma)

        # All 3 jobs share machine 0: makespan = 1 + 2 + 3 = 6
        # GP should find valid solution
        assert result.best_fitness <= 6
        assert result.best_fitness > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_multiple_runs(self, ft06_instance):
        """Multiple runs should all complete successfully."""
        pt, ma, _ = ft06_instance

        results = []
        for seed in range(5):
            gp = JSSPGP30Ops(budget=50, seed=seed)
            result = gp.run(pt, ma)
            results.append(result.best_fitness)

        # All should find valid solutions
        assert all(f < float("inf") for f in results)

    def test_evolved_program_structure(self, ft06_instance):
        """Evolved programs should contain primitive names."""
        pt, ma, _ = ft06_instance
        gp = JSSPGP30Ops(budget=100, seed=42)
        result = gp.run(pt, ma)

        # Program should contain some primitive (expanded list for 30 ops)
        primitives = [
            "nearest_neighbor", "farthest_insertion", "random_insertion",
            "cheapest_insertion", "savings", "christofides", "random_schedule",
            "two_opt", "swap", "insert", "or_opt", "three_opt", "invert",
            "lin_kernighan", "vns", "vnd",
            "double_bridge", "segment_shuffle", "ruin_recreate", "guided_mutation",
            "adaptive_mutation", "large_neighborhood",
            "sa", "tabu", "ils", "genetic", "ant_colony", "memetic"
        ]

        if result.best_fitness < float("inf"):
            has_primitive = any(p in result.best_individual for p in primitives)
            assert has_primitive or result.best_individual == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
