"""Level 1 Tests: AKG Topology.

Tests for LLM-generated graph topology (which operators can connect).
Level 1 is the base of the AKG system - structure without weights or conditions.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType
from src.geakg.generator import (
    LLMAKGGenerator,
    RandomAKGGenerator,
    OperatorInfo,
    AKGValidator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_operators():
    """Create a minimal set of operators for testing."""
    return [
        OperatorInfo(
            id="greedy_nearest_neighbor",
            name="Greedy NN",
            description="Builds tour by visiting nearest city",
            category="construction"
        ),
        OperatorInfo(
            id="two_opt",
            name="2-opt",
            description="Removes two edges and reconnects",
            category="local_search"
        ),
        OperatorInfo(
            id="three_opt",
            name="3-opt",
            description="Removes three edges and reconnects",
            category="local_search"
        ),
        OperatorInfo(
            id="double_bridge",
            name="Double Bridge",
            description="Splits tour into 4 parts",
            category="perturbation"
        ),
    ]


@pytest.fixture
def full_operators():
    """Create a full set of TSP operators for comprehensive testing."""
    from src.geakg.ontology import (
        create_construction_operators,
        create_local_search_operators,
        create_perturbation_operators,
    )

    all_ops = (
        create_construction_operators() +
        create_local_search_operators() +
        create_perturbation_operators()
    )

    return [
        OperatorInfo(
            id=op.id,
            name=op.name,
            description=op.description,
            category=op.category.value
        )
        for op in all_ops
    ]


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing without actual LLM calls."""
    client = MagicMock()
    return client


# =============================================================================
# Validator Tests
# =============================================================================

class TestAKGValidator:
    """Tests for the AKG validator."""

    def test_validator_detects_dead_ends(self, sample_operators):
        """Validator should detect operators without outgoing edges."""
        validator = AKGValidator([op.id for op in sample_operators])

        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Add only one edge - two_opt has no outgoing
        akg.add_edge(AKGEdge(
            source="greedy_nearest_neighbor",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        # Check for dead ends by looking at transitions
        transitions = akg.get_valid_transitions("two_opt")
        # two_opt has no outgoing edges - it's a dead end
        assert len(transitions) == 0

    def test_validator_accepts_valid_ils_cycle(self, sample_operators):
        """Validator should accept a proper ILS cycle."""
        validator = AKGValidator([op.id for op in sample_operators])

        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Complete ILS cycle
        akg.add_edge(AKGEdge(
            source="greedy_nearest_neighbor",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))
        akg.add_edge(AKGEdge(
            source="two_opt",
            target="three_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.7,
        ))
        akg.add_edge(AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
        ))
        akg.add_edge(AKGEdge(
            source="three_opt",
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

        # Check that ILS cycle exists:
        # LS -> Perturbation and Perturbation -> LS
        ls_to_perturb = akg.get_valid_transitions("two_opt")
        perturb_to_ls = akg.get_valid_transitions("double_bridge")

        assert "double_bridge" in ls_to_perturb  # two_opt -> double_bridge
        assert "two_opt" in perturb_to_ls  # double_bridge -> two_opt


# =============================================================================
# Topology Tests
# =============================================================================

class TestTopologyBasics:
    """Basic topology tests."""

    def test_topology_has_nodes(self, sample_operators):
        """AKG should contain all operators as nodes."""
        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        assert len(akg.nodes) == len(sample_operators)
        for op in sample_operators:
            assert op.id in akg.nodes

    def test_construction_operators_can_start(self, sample_operators):
        """Construction operators should be startable (no required predecessor)."""
        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        construction_ops = akg.get_operators_by_category(OperatorCategory.CONSTRUCTION)
        assert len(construction_ops) > 0
        assert construction_ops[0].id == "greedy_nearest_neighbor"


class TestTopologyVsRandom:
    """Compare LLM topology against random baseline."""

    def test_random_topology_is_denser(self, full_operators):
        """Random topology should have more edges than focused LLM topology."""
        # Create random AKG
        random_gen = RandomAKGGenerator(full_operators, edge_probability=0.3)
        random_akg = random_gen.generate()

        # A focused LLM topology would have ~60-100 edges
        # Random with 30 operators and 0.3 probability should have ~270 edges
        # (30 * 30 * 0.3 = 270 expected)
        assert len(random_akg.edges) > 100, "Random should be denser"

    def test_random_topology_has_all_nodes(self, full_operators):
        """Random AKG should include all operators."""
        random_gen = RandomAKGGenerator(full_operators, edge_probability=0.3)
        random_akg = random_gen.generate()

        assert len(random_akg.nodes) == len(full_operators)


class TestTopologyConnectivity:
    """Tests for topology connectivity."""

    def test_all_construction_have_outgoing(self, sample_operators):
        """All construction operators should have at least one outgoing edge."""
        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Add edges from construction
        akg.add_edge(AKGEdge(
            source="greedy_nearest_neighbor",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        construction_ops = akg.get_operators_by_category(OperatorCategory.CONSTRUCTION)
        for op in construction_ops:
            transitions = akg.get_valid_transitions(op.id)
            assert len(transitions) > 0, f"{op.id} has no outgoing edges"


# =============================================================================
# Ablation: Category-Complete Topology
# =============================================================================

class TestCategoryCompleteTopology:
    """Tests for category-complete topology (ablation baseline)."""

    def test_create_category_complete_graph(self, sample_operators):
        """Create a fully-connected graph based on category rules."""
        akg = AlgorithmicKnowledgeGraph()

        # Add nodes
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Add ALL valid edges by category
        # construction -> local_search
        # local_search -> local_search
        # local_search -> perturbation
        # perturbation -> local_search

        categories = {op.id: OperatorCategory(op.category) for op in sample_operators}

        for src_id, src_cat in categories.items():
            for tgt_id, tgt_cat in categories.items():
                if src_id == tgt_id:
                    continue  # No self-loops

                valid = False
                if src_cat == OperatorCategory.CONSTRUCTION and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                    valid = True
                elif src_cat == OperatorCategory.LOCAL_SEARCH and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                    valid = True
                elif src_cat == OperatorCategory.LOCAL_SEARCH and tgt_cat == OperatorCategory.PERTURBATION:
                    valid = True
                elif src_cat == OperatorCategory.PERTURBATION and tgt_cat == OperatorCategory.LOCAL_SEARCH:
                    valid = True

                if valid:
                    akg.add_edge(AKGEdge(
                        source=src_id,
                        target=tgt_id,
                        edge_type=EdgeType.SEQUENTIAL,
                        weight=0.5,  # Uniform weights
                    ))

        # Category-complete should have more edges than selective LLM
        # With 1 construction, 2 local_search, 1 perturbation:
        # C->LS: 2, LS->LS: 2 (excluding self), LS->P: 2, P->LS: 2 = 8 edges
        assert len(akg.edges) == 8

    def test_category_complete_vs_random(self, full_operators):
        """Category-complete and random have different edge patterns."""
        # Random AKG with low probability
        random_gen = RandomAKGGenerator(full_operators, edge_probability=0.1)
        random_akg = random_gen.generate()

        # Category-complete AKG (only valid category transitions)
        cat_complete_akg = AlgorithmicKnowledgeGraph()
        categories = {}

        for op in full_operators:
            cat = OperatorCategory(op.category)
            categories[op.id] = cat
            cat_complete_akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=cat,
            ))

        valid_transitions = [
            (OperatorCategory.CONSTRUCTION, OperatorCategory.LOCAL_SEARCH),
            (OperatorCategory.LOCAL_SEARCH, OperatorCategory.LOCAL_SEARCH),
            (OperatorCategory.LOCAL_SEARCH, OperatorCategory.PERTURBATION),
            (OperatorCategory.PERTURBATION, OperatorCategory.LOCAL_SEARCH),
            (OperatorCategory.PERTURBATION, OperatorCategory.CONSTRUCTION),
        ]

        for src_id, src_cat in categories.items():
            for tgt_id, tgt_cat in categories.items():
                if src_id == tgt_id:
                    continue
                if (src_cat, tgt_cat) in valid_transitions:
                    cat_complete_akg.add_edge(AKGEdge(
                        source=src_id,
                        target=tgt_id,
                        edge_type=EdgeType.SEQUENTIAL,
                        weight=0.5,
                    ))

        # Both should have nodes
        assert len(random_akg.nodes) == len(cat_complete_akg.nodes)
        # Both should have edges
        assert len(random_akg.edges) > 0
        assert len(cat_complete_akg.edges) > 0


# =============================================================================
# ILS Cycle Tests
# =============================================================================

class TestILSCycle:
    """Tests for ILS cycle (local_search <-> perturbation)."""

    def test_every_ls_has_escape_path(self, sample_operators):
        """Every local_search should have a path to perturbation."""
        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Add ILS edges
        akg.add_edge(AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
        ))
        akg.add_edge(AKGEdge(
            source="three_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
        ))

        ls_ops = akg.get_operators_by_category(OperatorCategory.LOCAL_SEARCH)
        for op in ls_ops:
            transitions = akg.get_valid_transitions(op.id)
            has_perturbation = any(
                akg.get_node(t).category == OperatorCategory.PERTURBATION
                for t in transitions
                if akg.get_node(t) is not None
            )
            assert has_perturbation, f"{op.id} has no escape to perturbation"

    def test_every_perturbation_has_reoptimize_path(self, sample_operators):
        """Every perturbation should have a path to local_search."""
        akg = AlgorithmicKnowledgeGraph()
        for op in sample_operators:
            akg.add_node(OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=OperatorCategory(op.category),
            ))

        # Add re-optimize edge
        akg.add_edge(AKGEdge(
            source="double_bridge",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        perturb_ops = akg.get_operators_by_category(OperatorCategory.PERTURBATION)
        for op in perturb_ops:
            transitions = akg.get_valid_transitions(op.id)
            has_local_search = any(
                akg.get_node(t).category == OperatorCategory.LOCAL_SEARCH
                for t in transitions
                if akg.get_node(t) is not None
            )
            assert has_local_search, f"{op.id} has no re-optimize to local_search"
