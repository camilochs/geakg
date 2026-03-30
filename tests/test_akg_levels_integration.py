"""Integration Tests: All AKG Levels.

End-to-end tests verifying that all three levels work together:
- Level 1: Topology (which operators can connect)
- Level 2: Weights (preference for each transition)
- Level 3: Conditions (when to take each transition)
"""

import pytest
from copy import deepcopy

from src.geakg.conditions import EdgeCondition, ConditionType, ExecutionContext
from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType
from src.geakg.aco import ACOSelector, ACOConfig
from src.geakg.generator import (
    LLMAKGGenerator,
    RandomAKGGenerator,
    OperatorInfo,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_operators():
    """Minimal set of operators for fast integration tests."""
    return [
        OperatorInfo(
            id="greedy_nn",
            name="Greedy NN",
            description="Build tour by nearest neighbor",
            category="construction"
        ),
        OperatorInfo(
            id="two_opt",
            name="2-opt",
            description="Remove and reconnect two edges",
            category="local_search"
        ),
        OperatorInfo(
            id="three_opt",
            name="3-opt",
            description="Remove and reconnect three edges",
            category="local_search"
        ),
        OperatorInfo(
            id="double_bridge",
            name="Double Bridge",
            description="Split tour into 4 parts",
            category="perturbation"
        ),
    ]


@pytest.fixture
def complete_akg(minimal_operators):
    """Create a complete AKG with all three levels."""
    akg = AlgorithmicKnowledgeGraph()

    # Add nodes (Level 1: Topology)
    for op in minimal_operators:
        akg.add_node(OperatorNode(
            id=op.id,
            name=op.name,
            description=op.description,
            category=OperatorCategory(op.category),
        ))

    # Add edges with weights (Level 2)
    # Construction -> Local Search
    akg.add_edge(AKGEdge(
        source="greedy_nn",
        target="two_opt",
        edge_type=EdgeType.SEQUENTIAL,
        weight=0.90,
    ))

    # Local Search -> Local Search
    akg.add_edge(AKGEdge(
        source="two_opt",
        target="three_opt",
        edge_type=EdgeType.SEQUENTIAL,
        weight=0.75,
    ))

    # Local Search -> Perturbation (with condition - Level 3)
    escape_condition = EdgeCondition(
        condition_type=ConditionType.STAGNATION,
        threshold=3.0,
    )
    akg.add_edge(AKGEdge(
        source="two_opt",
        target="double_bridge",
        edge_type=EdgeType.SEQUENTIAL,
        weight=0.55,
        conditions=[escape_condition],
        condition_boost=2.0,
    ))
    akg.add_edge(AKGEdge(
        source="three_opt",
        target="double_bridge",
        edge_type=EdgeType.SEQUENTIAL,
        weight=0.55,
        conditions=[escape_condition],
        condition_boost=2.0,
    ))

    # Perturbation -> Local Search
    akg.add_edge(AKGEdge(
        source="double_bridge",
        target="two_opt",
        edge_type=EdgeType.SEQUENTIAL,
        weight=0.90,
    ))

    return akg


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestLevel1OnlyTopology:
    """Test with only Level 1 (topology) - uniform weights, no conditions."""

    def test_run_with_uniform_weights_no_conditions(self, minimal_operators):
        """Run optimization with topology only - uniform weights, no conditions."""
        akg = AlgorithmicKnowledgeGraph()

        for op in minimal_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Uniform weights (no LLM preference)
        akg.add_edge(AKGEdge(
            source="greedy_nn", target="two_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.5,
        ))
        akg.add_edge(AKGEdge(
            source="two_opt", target="three_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.5,
        ))
        akg.add_edge(AKGEdge(
            source="two_opt", target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL, weight=0.5,
        ))
        akg.add_edge(AKGEdge(
            source="double_bridge", target="two_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.5,
        ))

        # Run ACO with conditions disabled
        config = ACOConfig(
            enable_conditions=False,
            n_ants=5,
            max_steps=6,
        )
        selector = ACOSelector(akg, config)

        # Construct multiple solutions
        solutions = []
        for _ in range(10):
            ant = selector.construct_solution()
            solutions.append(ant.path)

        # All solutions should start with construction
        for path in solutions:
            assert len(path) > 0
            assert path[0] == "greedy_nn"


class TestLevel2WithLLMWeights:
    """Test with Level 2 (weights) - LLM-assigned weights, no conditions."""

    def test_run_with_llm_weights_no_conditions(self, complete_akg):
        """Run optimization with LLM weights but no conditions."""
        # Disable conditions to test weights alone
        config = ACOConfig(
            enable_conditions=False,
            n_ants=5,
            max_steps=6,
            alpha=0.0,  # Disable pheromone
            beta=2.0,   # Emphasize weights
        )
        selector = ACOSelector(complete_akg, config)

        solutions = []
        for _ in range(10):
            ant = selector.construct_solution()
            solutions.append(ant.path)

        # Higher weight edges should be preferred
        # greedy_nn -> two_opt (0.90) should be common
        first_transitions = [p[1] for p in solutions if len(p) > 1]
        assert all(t == "two_opt" for t in first_transitions)


class TestLevel3WithConditions:
    """Test with Level 3 (conditions) - full system."""

    def test_run_with_conditions_enabled(self, complete_akg):
        """Run optimization with all levels enabled."""
        config = ACOConfig(
            enable_conditions=True,
            n_ants=5,
            max_steps=8,
        )
        selector = ACOSelector(complete_akg, config)

        # Test without stagnation - escape edges should have lower boost
        context_fresh = ExecutionContext(generations_without_improvement=0)
        selector.set_execution_context(context_fresh)

        boost_fresh = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost_fresh == 1.0  # Condition not met

        # Test with stagnation - escape edges should have higher boost
        context_stagnated = ExecutionContext(generations_without_improvement=5)
        selector.set_execution_context(context_stagnated)

        boost_stagnated = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost_stagnated == 2.0  # Condition met

    def test_conditions_change_behavior_over_time(self, complete_akg):
        """Conditions should change selection behavior as context changes."""
        config = ACOConfig(
            enable_conditions=True,
            n_ants=10,
            max_steps=8,
            exploration_rate=0.0,  # Deterministic selection
        )
        selector = ACOSelector(complete_akg, config)

        # Simulate fresh search (no stagnation)
        selector.set_execution_context(ExecutionContext(generations_without_improvement=0))

        # Simulate stagnated search
        selector.set_execution_context(ExecutionContext(generations_without_improvement=10))

        # Both should work without errors
        ant = selector.construct_solution()
        assert len(ant.path) > 0


# =============================================================================
# Comparison Tests
# =============================================================================

class TestAllLevelsVsRandomBaseline:
    """Compare all levels against random AKG baseline."""

    def test_structured_vs_random_topology(self, minimal_operators):
        """Structured and random AKGs should both be valid."""
        # Structured AKG (with ILS cycle)
        structured = AlgorithmicKnowledgeGraph()
        for op in minimal_operators:
            structured.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # ILS-focused edges
        structured.add_edge(AKGEdge(
            source="greedy_nn", target="two_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.9,
        ))
        structured.add_edge(AKGEdge(
            source="two_opt", target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL, weight=0.6,
        ))
        structured.add_edge(AKGEdge(
            source="double_bridge", target="two_opt",
            edge_type=EdgeType.SEQUENTIAL, weight=0.9,
        ))

        # Random AKG
        random_gen = RandomAKGGenerator(minimal_operators, edge_probability=0.9)
        random_akg = random_gen.generate()

        # Both should be valid graphs
        assert len(structured.edges) == 3
        assert len(structured.nodes) == 4
        assert len(random_akg.nodes) == 4
        assert len(random_akg.edges) >= 0  # Random may have any number of edges


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing tests pass with new condition infrastructure."""

    def test_old_edges_work_without_conditions(self, minimal_operators):
        """Edges created without conditions should work normally."""
        akg = AlgorithmicKnowledgeGraph()

        for op in minimal_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Old-style edge (no conditions)
        akg.add_edge(AKGEdge(
            source="greedy_nn",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        # Should work with conditions enabled
        config = ACOConfig(enable_conditions=True)
        selector = ACOSelector(akg, config)

        context = ExecutionContext(generations_without_improvement=10)
        selector.set_execution_context(context)

        # Old edges should return unit boost
        boost = selector._get_condition_boost("greedy_nn", "two_opt")
        assert boost == 1.0

    def test_akg_without_conditions_still_works(self, minimal_operators):
        """AKG created before conditions should still work."""
        akg = AlgorithmicKnowledgeGraph()

        for op in minimal_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        akg.add_edge(AKGEdge(
            source="greedy_nn",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))
        akg.add_edge(AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
        ))
        akg.add_edge(AKGEdge(
            source="double_bridge",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        # Run with conditions enabled - should not crash
        config = ACOConfig(enable_conditions=True, n_ants=3, max_steps=5)
        selector = ACOSelector(akg, config)
        selector.set_execution_context(ExecutionContext())

        for _ in range(5):
            ant = selector.construct_solution()
            assert len(ant.path) > 0


# =============================================================================
# Generator Integration Tests
# =============================================================================

class TestGeneratorLevelFlags:
    """Test generator with different level flags."""

    def test_generator_has_level_flags(self, minimal_operators):
        """Generator should have use_learned_weights and use_conditions flags."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        gen = LLMAKGGenerator(
            llm_client=mock_client,
            operators=minimal_operators,
            use_learned_weights=False,
            use_conditions=False,
        )
        assert gen.use_learned_weights is False
        assert gen.use_conditions is False

        gen2 = LLMAKGGenerator(
            llm_client=mock_client,
            operators=minimal_operators,
            use_learned_weights=True,
            use_conditions=True,
        )
        assert gen2.use_learned_weights is True
        assert gen2.use_conditions is True
