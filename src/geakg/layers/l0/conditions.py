"""Conditional Transitions for Algorithmic Knowledge Graphs.

Part of L1 (Unified Knowledge Layer): Adaptive control policies over algorithm composition.
Edges can have conditions that control WHEN a transition should be taken.

Example conditions:
- "After 3 generations without improvement" -> escape to perturbation
- "If gap > 5%" -> try aggressive local search
- "When diversity < 0.3" -> use perturbation

This provides adaptive control over the meta-algorithm topology.
"""

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class ConditionType(str, Enum):
    """Types of conditions supported on edges.

    Each condition type corresponds to a runtime metric that can be evaluated.
    """

    # Stagnation: generations without improvement
    STAGNATION = "stagnation"

    # Gap to best: (current - best) / best
    GAP_TO_BEST = "gap_to_best"

    # Diversity: population diversity (0-1)
    DIVERSITY_LOW = "diversity_low"

    # Improvement rate: fraction of recent decisions that improved
    IMPROVEMENT_RATE = "improvement_rate"

    # Budget remaining: (max - used) / max
    BUDGET_REMAINING = "budget_remaining"

    # Always: unconditional (default for backward compatibility)
    ALWAYS = "always"


class ComparisonOp(str, Enum):
    """Comparison operators for conditions."""

    GTE = "gte"  # >=
    LTE = "lte"  # <=
    GT = "gt"    # >
    LT = "lt"    # <
    EQ = "eq"    # ==


class EdgeCondition(BaseModel):
    """A condition that must be met for an edge transition.

    Conditions are evaluated against ExecutionContext at runtime.
    When multiple conditions exist on an edge, ALL must be satisfied (AND semantics).

    Example:
        EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            comparison=ComparisonOp.GTE,
            threshold=3.0,
            reason="Escape local optimum after 3 generations stuck"
        )
    """

    condition_type: ConditionType = ConditionType.ALWAYS
    comparison: ComparisonOp = ComparisonOp.GTE
    threshold: float = 0.0
    reason: str = ""  # Human-readable explanation

    model_config = {"frozen": True}

    def evaluate(self, context: "ExecutionContext") -> bool:
        """Evaluate this condition against runtime context.

        Args:
            context: Current execution state

        Returns:
            True if condition is satisfied, False otherwise
        """
        if self.condition_type == ConditionType.ALWAYS:
            return True

        value = context.get_metric(self.condition_type)
        if value is None:
            # Missing context = condition passes (permissive default)
            return True

        return self._compare(value, self.threshold)

    def _compare(self, value: float, threshold: float) -> bool:
        """Apply comparison operator."""
        match self.comparison:
            case ComparisonOp.GTE:
                return value >= threshold
            case ComparisonOp.LTE:
                return value <= threshold
            case ComparisonOp.GT:
                return value > threshold
            case ComparisonOp.LT:
                return value < threshold
            case ComparisonOp.EQ:
                return abs(value - threshold) < 1e-6
        return True


class ExecutionContext(BaseModel):
    """Runtime context for condition evaluation.

    Populated from the evolution engine's state and passed to ACO
    for condition evaluation during operator selection.

    Example:
        context = ExecutionContext(
            generations_without_improvement=5,
            current_fitness=25000.0,
            best_fitness=21282.0,
            population_diversity=0.25,
        )

        # Check if stagnation condition is met
        condition = EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=3.0
        )
        assert condition.evaluate(context) == True  # 5 >= 3
    """

    # Stagnation tracking
    generations_without_improvement: int = 0

    # Fitness metrics
    current_fitness: float = float("inf")
    best_fitness: float = float("inf")

    # Population metrics
    population_diversity: float = 0.5

    # Budget tracking
    evaluations_used: int = 0
    max_evaluations: int = 1000

    # Improvement tracking
    recent_improvement_rate: float = 0.0  # Fraction of last N ops that improved

    # Current algorithm state
    current_operators: list[str] = Field(default_factory=list)
    consecutive_local_search: int = 0

    # === Synthesis Metrics ===
    # Pheromone entropy for role transitions (0=converged, 1=uniform)
    pheromone_entropy: float = 1.0

    # Pheromone entropy for operators within roles (0=converged, 1=uniform)
    operator_pheromone_entropy: float = 1.0

    # Role usage frequencies in recent iterations (for bottleneck detection)
    recent_role_frequencies: dict[str, float] = Field(default_factory=dict)

    # Operator usage frequencies in recent iterations (for synthesis trigger)
    recent_operator_frequencies: dict[str, float] = Field(default_factory=dict)

    # Consecutive iterations with same best solution
    consecutive_same_best: int = 0

    def get_metric(self, condition_type: ConditionType) -> float | None:
        """Get the metric value for a condition type.

        Args:
            condition_type: The type of condition to get metric for

        Returns:
            The metric value, or None if not available
        """
        match condition_type:
            case ConditionType.STAGNATION:
                return float(self.generations_without_improvement)

            case ConditionType.GAP_TO_BEST:
                if self.best_fitness > 0 and self.best_fitness != float("inf"):
                    return (self.current_fitness - self.best_fitness) / self.best_fitness
                return 0.0

            case ConditionType.DIVERSITY_LOW:
                return self.population_diversity

            case ConditionType.BUDGET_REMAINING:
                if self.max_evaluations > 0:
                    return (self.max_evaluations - self.evaluations_used) / self.max_evaluations
                return 1.0

            case ConditionType.IMPROVEMENT_RATE:
                return self.recent_improvement_rate

            case ConditionType.ALWAYS:
                return 1.0

        return None

    @property
    def gap_to_best(self) -> float:
        """Calculate gap to best solution as percentage."""
        if self.best_fitness > 0 and self.best_fitness != float("inf"):
            return (self.current_fitness - self.best_fitness) / self.best_fitness
        return 0.0

    @property
    def is_stagnated(self) -> bool:
        """Check if search is stagnated (>3 generations without improvement)."""
        return self.generations_without_improvement > 3

    @property
    def budget_fraction_remaining(self) -> float:
        """Fraction of evaluation budget remaining."""
        if self.max_evaluations > 0:
            return (self.max_evaluations - self.evaluations_used) / self.max_evaluations
        return 1.0


def parse_condition_from_dict(data: dict) -> EdgeCondition:
    """Parse a condition from LLM output dictionary.

    Expected format:
        {"when": "stagnation", "threshold": 3, "comparison": "gte"}

    The "when" field maps to ConditionType, "comparison" is optional (defaults to gte).

    Args:
        data: Dictionary with condition specification

    Returns:
        EdgeCondition instance
    """
    # Map string to ConditionType
    type_map = {
        "stagnation": ConditionType.STAGNATION,
        "gap_to_best": ConditionType.GAP_TO_BEST,
        "gap": ConditionType.GAP_TO_BEST,  # Alias
        "diversity_low": ConditionType.DIVERSITY_LOW,
        "diversity": ConditionType.DIVERSITY_LOW,  # Alias
        "improvement_rate": ConditionType.IMPROVEMENT_RATE,
        "budget_remaining": ConditionType.BUDGET_REMAINING,
        "budget": ConditionType.BUDGET_REMAINING,  # Alias
        "always": ConditionType.ALWAYS,
    }

    # Map string to ComparisonOp
    comparison_map = {
        "gte": ComparisonOp.GTE,
        ">=": ComparisonOp.GTE,
        "lte": ComparisonOp.LTE,
        "<=": ComparisonOp.LTE,
        "gt": ComparisonOp.GT,
        ">": ComparisonOp.GT,
        "lt": ComparisonOp.LT,
        "<": ComparisonOp.LT,
        "eq": ComparisonOp.EQ,
        "==": ComparisonOp.EQ,
    }

    when = data.get("when", "always").lower()
    condition_type = type_map.get(when, ConditionType.ALWAYS)

    # Default comparison based on condition type
    default_comparison = ComparisonOp.GTE
    if condition_type == ConditionType.DIVERSITY_LOW:
        default_comparison = ComparisonOp.LTE  # diversity < threshold
    elif condition_type == ConditionType.IMPROVEMENT_RATE:
        default_comparison = ComparisonOp.LTE  # improvement_rate < threshold

    comparison_str = data.get("comparison", "").lower()
    comparison = comparison_map.get(comparison_str, default_comparison)

    threshold = float(data.get("threshold", 0.0))
    reason = data.get("reason", "")

    return EdgeCondition(
        condition_type=condition_type,
        comparison=comparison,
        threshold=threshold,
        reason=reason,
    )
