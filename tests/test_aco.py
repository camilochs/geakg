"""Tests for ACO-based traversal of Algorithmic Knowledge Graphs.

Verifies that all ACO components work correctly:
1. Pheromone initialization from LLM weights
2. Probabilistic operator selection
3. Energy budget constraints
4. Pheromone updates after evaluation
5. Solution construction
"""

import pytest
import random

from src.geakg.aco import ACOSelector, ACOConfig, Ant, GreedySelector
from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType


@pytest.fixture
def simple_akg():
    """Create a simple AKG for testing with 6 operators."""
    akg = AlgorithmicKnowledgeGraph()

    # Add construction operators
    akg.add_node(OperatorNode(
        id="nearest_neighbor",
        name="Nearest Neighbor",
        description="Greedy construction",
        category=OperatorCategory.CONSTRUCTION,
    ))
    akg.add_node(OperatorNode(
        id="random_init",
        name="Random Init",
        description="Random construction",
        category=OperatorCategory.CONSTRUCTION,
    ))

    # Add local search operators
    akg.add_node(OperatorNode(
        id="two_opt",
        name="2-Opt",
        description="2-opt improvement",
        category=OperatorCategory.LOCAL_SEARCH,
    ))
    akg.add_node(OperatorNode(
        id="three_opt",
        name="3-Opt",
        description="3-opt improvement",
        category=OperatorCategory.LOCAL_SEARCH,
    ))

    # Add perturbation operator
    akg.add_node(OperatorNode(
        id="double_bridge",
        name="Double Bridge",
        description="Perturbation move",
        category=OperatorCategory.PERTURBATION,
    ))

    # Add another perturbation operator (stronger)
    akg.add_node(OperatorNode(
        id="ruin_recreate",
        name="Ruin and Recreate",
        description="Strong perturbation",
        category=OperatorCategory.PERTURBATION,
    ))

    # Add edges with different weights (simulating LLM-assigned weights)
    # Construction -> Local Search (high weight)
    akg.add_edge(AKGEdge(
        source="nearest_neighbor", target="two_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.9
    ))
    akg.add_edge(AKGEdge(
        source="nearest_neighbor", target="three_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.7
    ))
    akg.add_edge(AKGEdge(
        source="random_init", target="two_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.8
    ))

    # Local Search -> Perturbation
    akg.add_edge(AKGEdge(
        source="two_opt", target="double_bridge",
        edge_type=EdgeType.SEQUENTIAL, weight=0.6
    ))
    akg.add_edge(AKGEdge(
        source="three_opt", target="double_bridge",
        edge_type=EdgeType.SEQUENTIAL, weight=0.5
    ))

    # Local Search -> Local Search (allow chaining)
    akg.add_edge(AKGEdge(
        source="two_opt", target="three_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.4
    ))

    # Perturbation -> Local Search (ILS pattern)
    akg.add_edge(AKGEdge(
        source="double_bridge", target="two_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.85
    ))

    # Strong perturbation connections
    akg.add_edge(AKGEdge(
        source="double_bridge", target="ruin_recreate",
        edge_type=EdgeType.SEQUENTIAL, weight=0.3
    ))
    akg.add_edge(AKGEdge(
        source="ruin_recreate", target="two_opt",
        edge_type=EdgeType.SEQUENTIAL, weight=0.7
    ))

    return akg


class TestACOConfig:
    """Test ACO configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ACOConfig()

        assert config.alpha == 1.0
        assert config.beta == 2.0
        assert config.rho == 0.1
        assert config.n_ants == 10
        assert config.max_steps == 8
        assert config.initial_energy == 9.0  # Energy for ~6 operators

    def test_custom_config(self):
        """Test custom configuration."""
        config = ACOConfig(
            alpha=2.0,
            beta=3.0,
            n_ants=20,
            max_steps=10,
        )

        assert config.alpha == 2.0
        assert config.beta == 3.0
        assert config.n_ants == 20
        assert config.max_steps == 10

    def test_energy_costs(self):
        """Test energy cost configuration."""
        config = ACOConfig()

        assert config.energy_costs["construction"] == 1.0
        assert config.energy_costs["local_search"] == 1.0
        assert config.energy_costs["perturbation"] == 1.0


class TestAnt:
    """Test Ant data structure."""

    def test_ant_initialization(self):
        """Test ant starts with empty path and full energy."""
        ant = Ant(energy=10.0)

        assert ant.path == []
        assert ant.energy == 10.0
        assert ant.fitness is None

    def test_can_continue(self):
        """Test energy-based continuation (max_steps ignored)."""
        ant = Ant(energy=5.0)
        ant.path = ["op1", "op2"]

        # Can continue with low cost
        assert ant.can_continue(cost=1.0, max_steps=8)

        # Cannot continue if cost exceeds energy
        assert not ant.can_continue(cost=6.0, max_steps=8)

        # max_steps is ignored - path length determined by energy only
        ant.path = ["op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8"]
        # Still can continue because energy (5.0) >= cost (1.0)
        assert ant.can_continue(cost=1.0, max_steps=8)

    def test_move_to(self):
        """Test ant movement consumes energy."""
        ant = Ant(energy=10.0)

        ant.move_to("op1", cost=1.5)
        assert ant.path == ["op1"]
        assert ant.energy == 8.5

        ant.move_to("op2", cost=2.0)
        assert ant.path == ["op1", "op2"]
        assert ant.energy == 6.5


class TestACOSelectorInitialization:
    """Test ACO selector initialization."""

    def test_pheromone_initialization(self, simple_akg):
        """Test pheromones are initialized from edge weights."""
        selector = ACOSelector(simple_akg)

        # Check pheromones match edge weights
        assert selector.pheromones[("nearest_neighbor", "two_opt")] == 0.9
        assert selector.pheromones[("nearest_neighbor", "three_opt")] == 0.7
        assert selector.pheromones[("two_opt", "double_bridge")] == 0.6

    def test_pheromone_minimum(self, simple_akg):
        """Test pheromones respect minimum value."""
        # Add edge with very low weight
        simple_akg.add_edge(AKGEdge(
            source="random_init", target="three_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.001
        ))

        config = ACOConfig(min_pheromone=0.05)
        selector = ACOSelector(simple_akg, config)

        # Should be clamped to minimum
        assert selector.pheromones[("random_init", "three_opt")] >= 0.05


class TestACOOperatorSelection:
    """Test probabilistic operator selection."""

    def test_valid_next_operators_from_start(self, simple_akg):
        """Test getting valid starting operators (construction only)."""
        selector = ACOSelector(simple_akg)

        valid = selector._get_valid_next_operators(None)

        assert "nearest_neighbor" in valid
        assert "random_init" in valid
        assert "two_opt" not in valid  # Not a construction operator

    def test_valid_next_operators_from_node(self, simple_akg):
        """Test getting valid transitions from a node."""
        selector = ACOSelector(simple_akg)

        valid = selector._get_valid_next_operators("nearest_neighbor")

        assert "two_opt" in valid
        assert "three_opt" in valid
        assert "double_bridge" not in valid  # No direct edge

    def test_heuristic_values(self, simple_akg):
        """Test heuristic values come from edge weights."""
        selector = ACOSelector(simple_akg)

        assert selector._get_heuristic("nearest_neighbor", "two_opt") == 0.9
        assert selector._get_heuristic("two_opt", "double_bridge") == 0.6

        # Non-existent edge returns minimum
        assert selector._get_heuristic("nearest_neighbor", "ils") == 0.01

    def test_energy_cost_by_category(self, simple_akg):
        """Test energy costs are uniform by default."""
        selector = ACOSelector(simple_akg)

        # All categories have cost 1.0 by default
        assert selector._get_energy_cost("nearest_neighbor") == 1.0
        assert selector._get_energy_cost("two_opt") == 1.0
        assert selector._get_energy_cost("double_bridge") == 1.0

    def test_probabilistic_selection_respects_weights(self, simple_akg):
        """Test that higher-weight edges are selected more often."""
        random.seed(42)
        config = ACOConfig(exploration_rate=0.0)  # No random exploration
        selector = ACOSelector(simple_akg, config)

        # Count selections from nearest_neighbor
        counts = {"two_opt": 0, "three_opt": 0}
        n_trials = 1000

        for _ in range(n_trials):
            ant = Ant(energy=10.0)
            selected = selector._select_next_operator(
                "nearest_neighbor", set(), ant
            )
            if selected in counts:
                counts[selected] += 1

        # two_opt (weight 0.9) should be selected more than three_opt (weight 0.7)
        assert counts["two_opt"] > counts["three_opt"]

        # With beta=2.0, the ratio should be roughly (0.9/0.7)^2 ≈ 1.65
        ratio = counts["two_opt"] / max(1, counts["three_opt"])
        assert ratio > 1.3  # Allow some variance

    def test_exploration_enables_random_selection(self, simple_akg):
        """Test exploration rate causes random selections."""
        random.seed(42)
        config = ACOConfig(exploration_rate=1.0)  # Always explore
        selector = ACOSelector(simple_akg, config)

        # With 100% exploration, selections should be more uniform
        counts = {"two_opt": 0, "three_opt": 0}
        n_trials = 1000

        for _ in range(n_trials):
            ant = Ant(energy=10.0)
            selected = selector._select_next_operator(
                "nearest_neighbor", set(), ant
            )
            if selected in counts:
                counts[selected] += 1

        # With random selection, ratio should be closer to 1
        ratio = counts["two_opt"] / max(1, counts["three_opt"])
        assert 0.7 < ratio < 1.5  # Roughly uniform


class TestACOSolutionConstruction:
    """Test solution (path) construction."""

    def test_solution_starts_with_construction(self, simple_akg):
        """Test solutions always start with construction operator."""
        random.seed(42)
        selector = ACOSelector(simple_akg)

        for _ in range(10):
            ant = selector.construct_solution()

            assert len(ant.path) > 0
            first_op = ant.path[0]
            node = simple_akg.get_node(first_op)
            assert node.category == OperatorCategory.CONSTRUCTION

    def test_solution_respects_energy_budget(self, simple_akg):
        """Test solutions don't exceed energy budget."""
        config = ACOConfig(initial_energy=3.0, max_steps=10, variable_energy=False)
        selector = ACOSelector(simple_akg, config)

        for _ in range(10):
            ant = selector.construct_solution()

            # Calculate total energy used
            total_cost = sum(
                selector._get_energy_cost(op) for op in ant.path
            )
            assert total_cost <= config.initial_energy

    def test_solution_length_determined_by_energy(self, simple_akg):
        """Test solution length is determined by energy budget, not max_steps."""
        # max_steps is ignored; path length depends on energy
        config = ACOConfig(max_steps=4, initial_energy=100.0)
        selector = ACOSelector(simple_akg, config)

        for _ in range(10):
            ant = selector.construct_solution()
            # Path can exceed max_steps since energy (100.0) allows it
            # Just verify path respects energy
            total_cost = sum(
                selector._get_energy_cost(op) for op in ant.path
            )
            assert total_cost <= config.initial_energy

    def test_solution_follows_valid_transitions(self, simple_akg):
        """Test each transition in solution is valid in AKG."""
        random.seed(42)
        selector = ACOSelector(simple_akg)

        for _ in range(10):
            ant = selector.construct_solution()

            for i in range(len(ant.path) - 1):
                source = ant.path[i]
                target = ant.path[i + 1]

                # Check edge exists in AKG
                valid_next = simple_akg.get_valid_transitions(source)
                assert target in valid_next, \
                    f"Invalid transition: {source} -> {target}"


class TestACOPheromoneUpdate:
    """Test pheromone updates."""

    def test_pheromone_evaporation(self, simple_akg):
        """Test pheromones evaporate after update."""
        config = ACOConfig(rho=0.2)  # 20% evaporation
        selector = ACOSelector(simple_akg, config)

        initial_pheromone = selector.pheromones[("nearest_neighbor", "two_opt")]

        # Update with empty ant list (only evaporation)
        selector.update_pheromones([])

        new_pheromone = selector.pheromones[("nearest_neighbor", "two_opt")]

        # Should be reduced by evaporation rate
        expected = initial_pheromone * (1 - config.rho)
        assert abs(new_pheromone - expected) < 0.01

    def test_pheromone_respects_maximum(self, simple_akg):
        """Test pheromones don't exceed maximum."""
        config = ACOConfig(max_pheromone=2.0, q=1000.0, rho=0.0)
        selector = ACOSelector(simple_akg, config)

        # Create very good ant
        ant = Ant()
        ant.path = ["nearest_neighbor", "two_opt"]
        ant.fitness = 1.0  # Extremely good

        # Multiple updates
        for _ in range(10):
            selector.update_pheromones([ant])

        # Should be clamped to max
        assert selector.pheromones[("nearest_neighbor", "two_opt")] <= config.max_pheromone

    def test_pheromone_respects_minimum(self, simple_akg):
        """Test pheromones don't go below minimum."""
        config = ACOConfig(min_pheromone=0.1, rho=0.5)  # Heavy evaporation
        selector = ACOSelector(simple_akg, config)

        # Many evaporation rounds
        for _ in range(20):
            selector.update_pheromones([])

        # All pheromones should respect minimum
        for pheromone in selector.pheromones.values():
            assert pheromone >= config.min_pheromone

    def test_best_solution_tracking(self, simple_akg):
        """Test best solution is tracked correctly."""
        selector = ACOSelector(simple_akg)

        # Initially no best
        assert selector.best_fitness == float("inf")
        assert selector.best_path == []

        # Good ant
        ant1 = Ant()
        ant1.path = ["nearest_neighbor", "two_opt"]
        ant1.fitness = 100.0

        selector.update_pheromones([ant1])
        assert selector.best_fitness == 100.0
        assert selector.best_path == ["nearest_neighbor", "two_opt"]

        # Better ant
        ant2 = Ant()
        ant2.path = ["random_init", "two_opt", "double_bridge"]
        ant2.fitness = 80.0

        selector.update_pheromones([ant2])
        assert selector.best_fitness == 80.0
        assert selector.best_path == ["random_init", "two_opt", "double_bridge"]

        # Worse ant doesn't update best
        ant3 = Ant()
        ant3.path = ["nearest_neighbor"]
        ant3.fitness = 150.0

        selector.update_pheromones([ant3])
        assert selector.best_fitness == 80.0  # Unchanged


class TestACOColonyRun:
    """Test full colony iteration."""

    def test_run_colony_produces_evaluated_ants(self, simple_akg):
        """Test run_colony produces ants with fitness values."""
        config = ACOConfig(n_ants=5)
        selector = ACOSelector(simple_akg, config)

        # Simple evaluate function
        def evaluate_fn(path, instance):
            return len(path) * 10.0  # Shorter paths = better

        ants = selector.run_colony(evaluate_fn, None)

        assert len(ants) > 0
        for ant in ants:
            assert ant.path  # Has a path
            assert ant.fitness is not None  # Has been evaluated

    def test_run_colony_updates_pheromones(self, simple_akg):
        """Test colony run updates pheromones."""
        config = ACOConfig(n_ants=10, rho=0.1)
        selector = ACOSelector(simple_akg, config)

        initial_pheromones = dict(selector.pheromones)

        def evaluate_fn(path, instance):
            return 50.0

        selector.run_colony(evaluate_fn, None)

        # Some pheromones should have changed
        changed = False
        for key in initial_pheromones:
            if abs(selector.pheromones[key] - initial_pheromones[key]) > 0.001:
                changed = True
                break

        assert changed, "Pheromones should change after colony run"

    def test_multiple_iterations_improve(self, simple_akg):
        """Test multiple iterations can find better solutions."""
        random.seed(42)
        config = ACOConfig(n_ants=10)
        selector = ACOSelector(simple_akg, config)

        # Reward specific patterns (construction -> local_search -> perturbation)
        def evaluate_fn(path, instance):
            score = 100.0
            if len(path) >= 2 and path[0] == "nearest_neighbor":
                score -= 20
            if len(path) >= 2 and path[1] == "two_opt":
                score -= 30
            if len(path) >= 3 and path[2] == "double_bridge":
                score -= 25
            return max(10, score)

        # Run multiple iterations
        for _ in range(10):
            selector.run_colony(evaluate_fn, None)

        # Best solution should include rewarded pattern
        best_path, best_fitness = selector.get_best_solution()
        assert best_fitness < 100.0, "Should find solutions better than baseline"


class TestGreedySelector:
    """Test greedy selector for comparison."""

    def test_greedy_selects_highest_weight(self, simple_akg):
        """Test greedy always selects highest weight transition."""
        selector = GreedySelector(simple_akg, max_steps=4)

        path = selector.construct_solution()

        # Should start with first construction op
        assert path[0] in ["nearest_neighbor", "random_init"]

        # From nearest_neighbor, should go to two_opt (weight 0.9 > 0.7)
        if path[0] == "nearest_neighbor":
            assert path[1] == "two_opt"

    def test_greedy_respects_max_steps(self, simple_akg):
        """Test greedy respects step limit."""
        selector = GreedySelector(simple_akg, max_steps=3)

        path = selector.construct_solution()

        assert len(path) <= 3

    def test_greedy_follows_valid_transitions(self, simple_akg):
        """Test greedy only follows valid edges."""
        selector = GreedySelector(simple_akg, max_steps=6)

        path = selector.construct_solution()

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            valid = simple_akg.get_valid_transitions(source)
            assert target in valid


class TestACOvsGreedy:
    """Compare ACO and Greedy behaviors."""

    def test_aco_explores_more_paths(self, simple_akg):
        """Test ACO explores more diverse paths than greedy."""
        random.seed(42)

        greedy = GreedySelector(simple_akg, max_steps=4)
        aco = ACOSelector(simple_akg, ACOConfig(n_ants=1, exploration_rate=0.3))

        # Greedy produces same path every time
        greedy_paths = set()
        for _ in range(20):
            path = greedy.construct_solution()
            greedy_paths.add(tuple(path))

        assert len(greedy_paths) == 1, "Greedy should always produce same path"

        # ACO explores different paths
        aco_paths = set()
        for _ in range(20):
            ant = aco.construct_solution()
            aco_paths.add(tuple(ant.path))

        assert len(aco_paths) > 1, "ACO should explore multiple paths"

    def test_aco_learns_from_experience(self, simple_akg):
        """Test ACO pheromones converge on good paths."""
        random.seed(42)
        config = ACOConfig(n_ants=5, rho=0.2)
        selector = ACOSelector(simple_akg, config)

        # Reward construction -> two_opt -> double_bridge pattern
        def evaluate_fn(path, instance):
            if (len(path) >= 3 and
                path[0] == "nearest_neighbor" and
                path[1] == "two_opt" and
                path[2] == "double_bridge"):
                return 10.0  # Very good
            return 100.0  # Baseline

        # Initial pheromones
        initial_nn_2opt = selector.pheromones[("nearest_neighbor", "two_opt")]
        initial_2opt_db = selector.pheromones[("two_opt", "double_bridge")]

        # Run many iterations
        for _ in range(20):
            selector.run_colony(evaluate_fn, None)

        # Pheromones on rewarded path should increase
        assert selector.pheromones[("nearest_neighbor", "two_opt")] > initial_nn_2opt
        assert selector.pheromones[("two_opt", "double_bridge")] > initial_2opt_db


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_akg(self):
        """Test handling of empty AKG."""
        akg = AlgorithmicKnowledgeGraph()
        selector = ACOSelector(akg)

        ant = selector.construct_solution()
        assert ant.path == []

    def test_no_outgoing_edges(self):
        """Test handling of nodes with no outgoing edges."""
        akg = AlgorithmicKnowledgeGraph()

        # Add isolated construction operator
        akg.add_node(OperatorNode(
            id="lonely_op",
            name="Lonely Op",
            description="No connections",
            category=OperatorCategory.CONSTRUCTION,
        ))

        selector = ACOSelector(akg)
        ant = selector.construct_solution()

        # Should just have the construction operator
        assert ant.path == ["lonely_op"]

    def test_evaluation_failure_handling(self, simple_akg):
        """Test handling of evaluation failures."""
        selector = ACOSelector(simple_akg)

        def failing_evaluate(path, instance):
            raise ValueError("Evaluation failed!")

        # Should not crash, just mark as infinite fitness
        ants = selector.run_colony(failing_evaluate, None)

        for ant in ants:
            assert ant.fitness == float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
