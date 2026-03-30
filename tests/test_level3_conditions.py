"""Level 3 Tests: Conditional Transitions.

Tests for edge conditions that control WHEN transitions should be taken.
Level 3 adds control policies on top of Level 1 (topology) and Level 2 (weights).

The LLM induces control policies over algorithm composition:
- "After 3 generations without improvement" → escape to perturbation
- "If gap > 5%" → try aggressive local search
"""

import pytest

from src.geakg.conditions import (
    ConditionType,
    ComparisonOp,
    EdgeCondition,
    ExecutionContext,
    parse_condition_from_dict,
)
from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType
from src.geakg.aco import ACOSelector, ACOConfig


# =============================================================================
# EdgeCondition Unit Tests
# =============================================================================

class TestEdgeConditionBasics:
    """Basic tests for EdgeCondition class."""

    def test_condition_always_passes(self):
        """ALWAYS condition should always return True."""
        condition = EdgeCondition(
            condition_type=ConditionType.ALWAYS,
        )
        context = ExecutionContext()
        assert condition.evaluate(context) is True

    def test_condition_stagnation_triggers(self):
        """Stagnation condition should trigger after threshold."""
        condition = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            comparison=ComparisonOp.GTE,
            threshold=3.0,
        )

        # Not stagnated yet
        context = ExecutionContext(generations_without_improvement=2)
        assert condition.evaluate(context) is False

        # Exactly at threshold
        context = ExecutionContext(generations_without_improvement=3)
        assert condition.evaluate(context) is True

        # Beyond threshold
        context = ExecutionContext(generations_without_improvement=5)
        assert condition.evaluate(context) is True

    def test_condition_gap_to_best_triggers(self):
        """Gap condition should trigger when gap exceeds threshold."""
        condition = EdgeCondition(
            condition_type=ConditionType.GAP_TO_BEST,
            comparison=ComparisonOp.GTE,
            threshold=0.05,  # 5%
        )

        # Small gap (3%)
        context = ExecutionContext(current_fitness=21900, best_fitness=21282)
        assert condition.evaluate(context) is False  # 2.9%

        # Large gap (10%)
        context = ExecutionContext(current_fitness=23400, best_fitness=21282)
        assert condition.evaluate(context) is True  # 10%

    def test_condition_diversity_low_triggers(self):
        """Diversity condition should trigger when below threshold."""
        condition = EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            comparison=ComparisonOp.LTE,
            threshold=0.3,
        )

        # High diversity
        context = ExecutionContext(population_diversity=0.5)
        assert condition.evaluate(context) is False

        # Low diversity
        context = ExecutionContext(population_diversity=0.2)
        assert condition.evaluate(context) is True

    def test_condition_improvement_rate_triggers(self):
        """Improvement rate condition should trigger when below threshold."""
        condition = EdgeCondition(
            condition_type=ConditionType.IMPROVEMENT_RATE,
            comparison=ComparisonOp.LTE,
            threshold=0.3,
        )

        # Good improvement rate
        context = ExecutionContext(recent_improvement_rate=0.6)
        assert condition.evaluate(context) is False

        # Poor improvement rate
        context = ExecutionContext(recent_improvement_rate=0.2)
        assert condition.evaluate(context) is True


class TestComparisonOperators:
    """Tests for different comparison operators."""

    def test_gte_comparison(self):
        """GTE comparison should work correctly."""
        condition = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            comparison=ComparisonOp.GTE,
            threshold=5.0,
        )

        assert condition.evaluate(ExecutionContext(generations_without_improvement=4)) is False
        assert condition.evaluate(ExecutionContext(generations_without_improvement=5)) is True
        assert condition.evaluate(ExecutionContext(generations_without_improvement=6)) is True

    def test_lte_comparison(self):
        """LTE comparison should work correctly."""
        condition = EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            comparison=ComparisonOp.LTE,
            threshold=0.3,
        )

        assert condition.evaluate(ExecutionContext(population_diversity=0.4)) is False
        assert condition.evaluate(ExecutionContext(population_diversity=0.3)) is True
        assert condition.evaluate(ExecutionContext(population_diversity=0.2)) is True

    def test_gt_comparison(self):
        """GT comparison should work correctly."""
        condition = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            comparison=ComparisonOp.GT,
            threshold=5.0,
        )

        assert condition.evaluate(ExecutionContext(generations_without_improvement=5)) is False
        assert condition.evaluate(ExecutionContext(generations_without_improvement=6)) is True

    def test_lt_comparison(self):
        """LT comparison should work correctly."""
        condition = EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            comparison=ComparisonOp.LT,
            threshold=0.3,
        )

        assert condition.evaluate(ExecutionContext(population_diversity=0.3)) is False
        assert condition.evaluate(ExecutionContext(population_diversity=0.29)) is True


# =============================================================================
# ExecutionContext Tests
# =============================================================================

class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_gap_calculation(self):
        """Gap to best should be calculated correctly."""
        context = ExecutionContext(
            current_fitness=23400,
            best_fitness=21282,
        )
        expected_gap = (23400 - 21282) / 21282  # ~0.0996
        assert abs(context.gap_to_best - expected_gap) < 0.001

    def test_gap_calculation_zero_best(self):
        """Gap should be 0 when best fitness is 0 or inf."""
        context = ExecutionContext(
            current_fitness=100,
            best_fitness=0,
        )
        assert context.gap_to_best == 0.0

        context = ExecutionContext(
            current_fitness=100,
            best_fitness=float("inf"),
        )
        assert context.gap_to_best == 0.0

    def test_is_stagnated_property(self):
        """is_stagnated should be True when >3 generations stuck."""
        context = ExecutionContext(generations_without_improvement=3)
        assert context.is_stagnated is False

        context = ExecutionContext(generations_without_improvement=4)
        assert context.is_stagnated is True

    def test_budget_fraction_remaining(self):
        """Budget fraction should be calculated correctly."""
        context = ExecutionContext(
            evaluations_used=300,
            max_evaluations=1000,
        )
        assert context.budget_fraction_remaining == 0.7

    def test_get_metric_for_all_types(self):
        """get_metric should work for all condition types."""
        context = ExecutionContext(
            generations_without_improvement=5,
            current_fitness=25000,
            best_fitness=21282,
            population_diversity=0.4,
            evaluations_used=500,
            max_evaluations=1000,
            recent_improvement_rate=0.3,
        )

        assert context.get_metric(ConditionType.STAGNATION) == 5.0
        assert context.get_metric(ConditionType.DIVERSITY_LOW) == 0.4
        assert context.get_metric(ConditionType.BUDGET_REMAINING) == 0.5
        assert context.get_metric(ConditionType.IMPROVEMENT_RATE) == 0.3
        assert context.get_metric(ConditionType.ALWAYS) == 1.0

        gap = context.get_metric(ConditionType.GAP_TO_BEST)
        expected = (25000 - 21282) / 21282
        assert abs(gap - expected) < 0.001


# =============================================================================
# Condition Parsing Tests
# =============================================================================

class TestConditionParsing:
    """Tests for parsing conditions from LLM output."""

    def test_parse_stagnation_condition(self):
        """Parse stagnation condition from dict."""
        data = {"when": "stagnation", "threshold": 3}
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.STAGNATION
        assert condition.threshold == 3.0
        assert condition.comparison == ComparisonOp.GTE  # Default for stagnation

    def test_parse_gap_condition(self):
        """Parse gap_to_best condition from dict."""
        data = {"when": "gap_to_best", "threshold": 0.05}
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.GAP_TO_BEST
        assert condition.threshold == 0.05

    def test_parse_with_alias(self):
        """Parse condition with alias names."""
        data = {"when": "gap", "threshold": 0.1}  # "gap" is alias for "gap_to_best"
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.GAP_TO_BEST

        data = {"when": "diversity", "threshold": 0.3}
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.DIVERSITY_LOW

    def test_parse_with_custom_comparison(self):
        """Parse condition with explicit comparison operator."""
        data = {"when": "stagnation", "threshold": 5, "comparison": "gt"}
        condition = parse_condition_from_dict(data)

        assert condition.comparison == ComparisonOp.GT

    def test_parse_invalid_type_falls_back_to_always(self):
        """Invalid condition type should fall back to ALWAYS."""
        data = {"when": "unknown_type", "threshold": 5}
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.ALWAYS

    def test_parse_empty_dict(self):
        """Empty dict should result in ALWAYS condition."""
        data = {}
        condition = parse_condition_from_dict(data)

        assert condition.condition_type == ConditionType.ALWAYS


# =============================================================================
# Edge with Conditions Tests
# =============================================================================

class TestEdgeWithConditions:
    """Tests for AKGEdge with conditions."""

    def test_edge_without_conditions_backward_compat(self):
        """Edge without conditions should always evaluate to (True, 1.0)."""
        edge = AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
            # No conditions
        )

        context = ExecutionContext(generations_without_improvement=10)
        satisfied, boost = edge.evaluate_conditions(context)

        assert satisfied is True
        assert boost == 1.0

    def test_edge_with_conditions_evaluates(self):
        """Edge with conditions should evaluate correctly."""
        stagnation_condition = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            comparison=ComparisonOp.GTE,
            threshold=3.0,
        )

        edge = AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
            conditions=[stagnation_condition],
            condition_boost=2.0,
        )

        # Not stagnated - condition not met
        context = ExecutionContext(generations_without_improvement=2)
        satisfied, boost = edge.evaluate_conditions(context)
        assert satisfied is False
        assert boost == 1.0  # No boost when condition not met

        # Stagnated - condition met
        context = ExecutionContext(generations_without_improvement=5)
        satisfied, boost = edge.evaluate_conditions(context)
        assert satisfied is True
        assert boost == 2.0

    def test_edge_with_multiple_conditions_all_must_pass(self):
        """Multiple conditions should use AND semantics."""
        stagnation = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=3.0,
        )
        low_diversity = EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            comparison=ComparisonOp.LTE,
            threshold=0.3,
        )

        edge = AKGEdge(
            source="two_opt",
            target="ruin_recreate",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.5,
            conditions=[stagnation, low_diversity],
            condition_boost=3.0,
        )

        # Only stagnation met
        context = ExecutionContext(
            generations_without_improvement=5,
            population_diversity=0.5,
        )
        satisfied, boost = edge.evaluate_conditions(context)
        assert satisfied is False  # Both must be met

        # Both conditions met
        context = ExecutionContext(
            generations_without_improvement=5,
            population_diversity=0.2,
        )
        satisfied, boost = edge.evaluate_conditions(context)
        assert satisfied is True
        assert boost == 3.0

    def test_has_conditions_method(self):
        """has_conditions should return correct boolean."""
        edge_no_cond = AKGEdge(
            source="a",
            target="b",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.5,
        )
        assert edge_no_cond.has_conditions() is False

        edge_with_cond = AKGEdge(
            source="a",
            target="b",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.5,
            conditions=[EdgeCondition(condition_type=ConditionType.STAGNATION, threshold=3)],
        )
        assert edge_with_cond.has_conditions() is True


# =============================================================================
# ACO with Conditions Tests
# =============================================================================

class TestACOWithConditions:
    """Tests for ACO integration with conditions."""

    @pytest.fixture
    def akg_with_conditions(self):
        """Create an AKG with conditional edges."""
        akg = AlgorithmicKnowledgeGraph()

        operators = [
            ("construction", "Construction", OperatorCategory.CONSTRUCTION),
            ("two_opt", "2-opt", OperatorCategory.LOCAL_SEARCH),
            ("double_bridge", "Double Bridge", OperatorCategory.PERTURBATION),
        ]

        for op_id, name, category in operators:
            akg.add_node(OperatorNode(
                id=op_id,
                name=name,
                description=f"Test {name}",
                category=category,
            ))

        # Normal edge (no condition)
        akg.add_edge(AKGEdge(
            source="construction",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        # Conditional escape edge
        stagnation_cond = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=3.0,
        )
        akg.add_edge(AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.5,
            conditions=[stagnation_cond],
            condition_boost=2.0,
        ))

        # Re-optimize edge
        akg.add_edge(AKGEdge(
            source="double_bridge",
            target="two_opt",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.9,
        ))

        return akg

    def test_conditions_disabled_ignores_conditions(self, akg_with_conditions):
        """With enable_conditions=False, conditions should be ignored."""
        config = ACOConfig(enable_conditions=False)
        selector = ACOSelector(akg_with_conditions, config)

        # Set stagnation context
        context = ExecutionContext(generations_without_improvement=5)
        selector.set_execution_context(context)

        # Get boost - should be 1.0 since conditions disabled
        boost = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost == 1.0

    def test_conditions_enabled_applies_boost(self, akg_with_conditions):
        """With enable_conditions=True, boost should be applied when condition met."""
        config = ACOConfig(enable_conditions=True)
        selector = ACOSelector(akg_with_conditions, config)

        # Set stagnation context (condition met)
        context = ExecutionContext(generations_without_improvement=5)
        selector.set_execution_context(context)

        boost = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost == 2.0

    def test_no_context_returns_unit_boost(self, akg_with_conditions):
        """Without execution context, should return 1.0 boost."""
        config = ACOConfig(enable_conditions=True)
        selector = ACOSelector(akg_with_conditions, config)

        # No context set
        boost = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost == 1.0

    def test_condition_not_met_returns_unit_boost(self, akg_with_conditions):
        """When condition not met, should return 1.0 boost."""
        config = ACOConfig(enable_conditions=True)
        selector = ACOSelector(akg_with_conditions, config)

        # Not stagnated (condition not met)
        context = ExecutionContext(generations_without_improvement=2)
        selector.set_execution_context(context)

        boost = selector._get_condition_boost("two_opt", "double_bridge")
        assert boost == 1.0


# =============================================================================
# Ablation Tests
# =============================================================================

class TestConditionsAblation:
    """Tests for conditions ablation (E14 experiment)."""

    def test_ablation_flag_works(self):
        """enable_conditions flag should control condition evaluation."""
        config_enabled = ACOConfig(enable_conditions=True)
        config_disabled = ACOConfig(enable_conditions=False)

        assert config_enabled.enable_conditions is True
        assert config_disabled.enable_conditions is False

    def test_same_akg_different_ablation(self):
        """Same AKG should behave differently with ablation flag."""
        akg = AlgorithmicKnowledgeGraph()

        akg.add_node(OperatorNode(
            id="a",
            name="A",
            description="A",
            category=OperatorCategory.CONSTRUCTION,
        ))
        akg.add_node(OperatorNode(
            id="b",
            name="B",
            description="B",
            category=OperatorCategory.LOCAL_SEARCH,
        ))

        stag = EdgeCondition(condition_type=ConditionType.STAGNATION, threshold=3)
        akg.add_edge(AKGEdge(
            source="a",
            target="b",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.5,
            conditions=[stag],
            condition_boost=2.0,
        ))

        # With conditions enabled
        selector_enabled = ACOSelector(akg, ACOConfig(enable_conditions=True))
        selector_enabled.set_execution_context(ExecutionContext(generations_without_improvement=5))
        boost_enabled = selector_enabled._get_condition_boost("a", "b")

        # With conditions disabled
        selector_disabled = ACOSelector(akg, ACOConfig(enable_conditions=False))
        selector_disabled.set_execution_context(ExecutionContext(generations_without_improvement=5))
        boost_disabled = selector_disabled._get_condition_boost("a", "b")

        assert boost_enabled == 2.0
        assert boost_disabled == 1.0
