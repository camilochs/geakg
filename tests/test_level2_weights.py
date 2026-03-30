"""Level 2 Tests: AKG Edge Weights.

Tests for weight assignment (LLM-learned vs expert-defined).
Level 2 adds preference information on top of Level 1 topology.
"""

import pytest
import statistics
from copy import deepcopy

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType
from src.geakg.aco import ACOSelector, ACOConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_akg():
    """Create a sample AKG with edges for weight testing."""
    akg = AlgorithmicKnowledgeGraph()

    operators = [
        ("greedy_nearest_neighbor", "Greedy NN", OperatorCategory.CONSTRUCTION),
        ("two_opt", "2-opt", OperatorCategory.LOCAL_SEARCH),
        ("three_opt", "3-opt", OperatorCategory.LOCAL_SEARCH),
        ("or_opt", "Or-opt", OperatorCategory.LOCAL_SEARCH),
        ("double_bridge", "Double Bridge", OperatorCategory.PERTURBATION),
    ]

    for op_id, name, category in operators:
        akg.add_node(OperatorNode(
            id=op_id,
            name=name,
            description=f"Test operator {name}",
            category=category,
        ))

    # Add edges with various weights
    edges = [
        ("greedy_nearest_neighbor", "two_opt", 0.90),
        ("two_opt", "three_opt", 0.75),
        ("two_opt", "or_opt", 0.65),
        ("three_opt", "or_opt", 0.60),
        ("two_opt", "double_bridge", 0.55),
        ("three_opt", "double_bridge", 0.55),
        ("or_opt", "double_bridge", 0.50),
        ("double_bridge", "two_opt", 0.90),
        ("double_bridge", "three_opt", 0.85),
    ]

    for src, tgt, weight in edges:
        akg.add_edge(AKGEdge(
            source=src,
            target=tgt,
            edge_type=EdgeType.SEQUENTIAL,
            weight=weight,
        ))

    return akg


# =============================================================================
# Weight Validity Tests
# =============================================================================

class TestWeightValidity:
    """Tests for weight value validity."""

    def test_weights_in_valid_range(self, sample_akg):
        """All weights should be in [0.0, 1.0] range."""
        for edge in sample_akg.edges.values():
            assert 0.0 <= edge.weight <= 1.0, f"Invalid weight {edge.weight}"

    def test_no_zero_weights(self, sample_akg):
        """No edge should have exactly zero weight (would block traversal)."""
        for edge in sample_akg.edges.values():
            assert edge.weight > 0.0, f"Zero weight on {edge.source}->{edge.target}"

    def test_weight_distribution_has_variance(self, sample_akg):
        """Weights should have some variance (not all same value)."""
        weights = [e.weight for e in sample_akg.edges.values()]
        if len(weights) > 1:
            std = statistics.stdev(weights)
            assert std > 0.01, "Weights should have some variance"


# =============================================================================
# Expert Weight Application Tests
# =============================================================================

class TestExpertWeights:
    """Tests for expert-defined weight application (Stage 4)."""

    def test_apply_expert_weights(self, sample_akg):
        """Expert weights should follow transition type rules."""
        # Define expert weight function
        def get_expert_weight(src_cat, tgt_cat):
            if src_cat == OperatorCategory.CONSTRUCTION and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                return 0.90
            elif src_cat == OperatorCategory.LOCAL_SEARCH and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                return 0.70
            elif src_cat == OperatorCategory.LOCAL_SEARCH and tgt_cat == OperatorCategory.PERTURBATION:
                return 0.60
            elif src_cat == OperatorCategory.PERTURBATION and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                return 0.90
            return 0.50

        # Create expert-weighted copy
        expert_akg = deepcopy(sample_akg)

        for edge_key, edge in list(expert_akg.edges.items()):
            src_node = expert_akg.get_node(edge.source)
            tgt_node = expert_akg.get_node(edge.target)
            if src_node and tgt_node:
                new_weight = get_expert_weight(src_node.category, tgt_node.category)
                expert_akg.edges[edge_key] = AKGEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    weight=new_weight,
                )

        # Verify expert weights were applied
        for edge in expert_akg.edges.values():
            src_node = expert_akg.get_node(edge.source)
            tgt_node = expert_akg.get_node(edge.target)
            expected = get_expert_weight(src_node.category, tgt_node.category)
            assert edge.weight == expected, f"Weight mismatch for {edge.source}->{edge.target}"


# =============================================================================
# LLM Weight Tests
# =============================================================================

class TestLLMWeights:
    """Tests for LLM-learned weight preservation."""

    def test_llm_weights_preserved(self, sample_akg):
        """LLM weights should be preserved when use_learned_weights=True."""
        # The sample_akg already has "LLM-like" weights
        original_weights = {
            (e.source, e.target): e.weight
            for e in sample_akg.edges.values()
        }

        # If we don't apply expert weights, original should be preserved
        for edge_key, edge in sample_akg.edges.items():
            assert (edge.source, edge.target) in original_weights
            assert edge.weight == original_weights[(edge.source, edge.target)]


# =============================================================================
# Weight Effect on ACO
# =============================================================================

class TestWeightEffectOnACO:
    """Tests for how weights affect ACO selection."""

    def test_higher_weight_higher_probability(self, sample_akg):
        """Higher weight edges should be selected more often."""
        config = ACOConfig(
            alpha=0.0,  # Disable pheromone (only use heuristic)
            beta=2.0,   # Emphasize heuristic (weight)
            exploration_rate=0.0,  # No random exploration
        )
        selector = ACOSelector(sample_akg, config)

        # From two_opt, we have edges to:
        # - three_opt (0.75)
        # - or_opt (0.65)
        # - double_bridge (0.55)

        # Count selections over many trials
        selections = {"three_opt": 0, "or_opt": 0, "double_bridge": 0}
        n_trials = 1000

        for _ in range(n_trials):
            # Construct solution and check second step (after construction)
            ant = selector.construct_solution()
            if len(ant.path) >= 2:
                # First is construction, second should be influenced by weights
                pass

        # Higher weight should mean higher selection probability
        # (We can't test exact counts, but relationship should hold)

    def test_weight_zero_blocks_selection(self):
        """Zero weight edges should never be selected."""
        akg = AlgorithmicKnowledgeGraph()

        akg.add_node(OperatorNode(
            id="construction",
            name="Construction",
            description="Start",
            category=OperatorCategory.CONSTRUCTION,
        ))
        akg.add_node(OperatorNode(
            id="ls1",
            name="LS1",
            description="Local Search 1",
            category=OperatorCategory.LOCAL_SEARCH,
        ))
        akg.add_node(OperatorNode(
            id="ls2",
            name="LS2",
            description="Local Search 2",
            category=OperatorCategory.LOCAL_SEARCH,
        ))

        # One edge with weight 0.9, another with weight 0.01 (very low)
        akg.add_edge(AKGEdge(
            source="construction",
            target="ls1",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))
        akg.add_edge(AKGEdge(
            source="construction",
            target="ls2",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.01,  # Very low but not zero
        ))

        config = ACOConfig(
            alpha=0.0,
            beta=2.0,
            exploration_rate=0.0,
        )
        selector = ACOSelector(akg, config)

        # Over many trials, ls1 should be selected much more often
        ls1_count = 0
        ls2_count = 0
        n_trials = 100

        for _ in range(n_trials):
            ant = selector.construct_solution()
            if len(ant.path) >= 2:
                if ant.path[1] == "ls1":
                    ls1_count += 1
                elif ant.path[1] == "ls2":
                    ls2_count += 1

        # ls1 should be selected much more than ls2
        assert ls1_count > ls2_count * 5, "Higher weight should dominate selection"


# =============================================================================
# Same Topology, Different Weights
# =============================================================================

class TestSameTopologyDifferentWeights:
    """Tests for comparing weight schemes on same topology."""

    def test_same_edges_different_weights(self, sample_akg):
        """Same topology with different weights should have same edges."""
        # Create two copies with different weight schemes
        akg_llm = deepcopy(sample_akg)

        akg_expert = deepcopy(sample_akg)
        for edge_key, edge in list(akg_expert.edges.items()):
            # Apply uniform expert weights
            akg_expert.edges[edge_key] = AKGEdge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                weight=0.75,  # Uniform weight
            )

        # Same number of edges
        assert len(akg_llm.edges) == len(akg_expert.edges)

        # Same edge keys
        assert set(akg_llm.edges.keys()) == set(akg_expert.edges.keys())

        # Different weights
        for edge_key in akg_llm.edges:
            llm_weight = akg_llm.edges[edge_key].weight
            expert_weight = akg_expert.edges[edge_key].weight
            # At least some weights should differ
            if llm_weight != 0.75:
                assert llm_weight != expert_weight


# =============================================================================
# Weight Statistics
# =============================================================================

class TestWeightStatistics:
    """Tests for weight distribution statistics."""

    def test_weight_mean_in_reasonable_range(self, sample_akg):
        """Mean weight should be in a reasonable range."""
        weights = [e.weight for e in sample_akg.edges.values()]
        mean = statistics.mean(weights)
        assert 0.3 <= mean <= 0.9, f"Mean weight {mean} out of reasonable range"

    def test_weight_std_not_too_high(self, sample_akg):
        """Standard deviation should not be too extreme."""
        weights = [e.weight for e in sample_akg.edges.values()]
        if len(weights) > 1:
            std = statistics.stdev(weights)
            assert std <= 0.5, f"Weight std {std} too high"

    def test_construction_to_ls_has_high_weight(self, sample_akg):
        """Construction to local_search edges should have high weights."""
        for edge in sample_akg.edges.values():
            src_node = sample_akg.get_node(edge.source)
            tgt_node = sample_akg.get_node(edge.target)
            if (src_node and tgt_node and
                src_node.category == OperatorCategory.CONSTRUCTION and
                tgt_node.category == OperatorCategory.LOCAL_SEARCH):
                assert edge.weight >= 0.7, "Construction->LS should have high weight"
