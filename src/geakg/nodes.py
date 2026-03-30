"""Node definitions for the Algorithmic Knowledge Graph."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.geakg.conditions import EdgeCondition, ExecutionContext


class NodeType(str, Enum):
    """Types of nodes in the AKG."""

    OPERATOR = "operator"
    DATA_STRUCTURE = "data_structure"
    PROPERTY = "property"


class OperatorCategory(str, Enum):
    """Categories of operator nodes.

    Core categories for optimization (3) plus extensible categories
    for other case studies (NAS).
    """

    CONSTRUCTION = "construction"
    LOCAL_SEARCH = "local_search"
    PERTURBATION = "perturbation"

    # NAS categories
    TOPOLOGY = "topology"
    ACTIVATION = "activation"
    TRAINING = "training"
    REGULARIZATION = "regularization"
    EVALUATION = "evaluation"


class EdgeType(str, Enum):
    """Types of edges in the AKG."""

    SEQUENTIAL = "sequential"  # A can follow B in a pipeline
    COMPATIBLE = "compatible"  # A and B can be used together
    REQUIRES = "requires"  # A requires B as prerequisite
    IMPROVES = "improves"  # A typically improves solutions from B


class AKGNode(BaseModel):
    """Base class for all AKG nodes."""

    id: str
    name: str
    node_type: NodeType
    description: str = ""

    model_config = {"frozen": True}


class OperatorNode(AKGNode):
    """Node representing a heuristic operator.

    Operators are the core building blocks of algorithms in the AKG.
    They transform solutions (construction) or improve existing ones (local search).
    """

    node_type: NodeType = NodeType.OPERATOR
    category: OperatorCategory
    preconditions: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    code_template: str | None = None
    domains: list[str] = Field(default_factory=lambda: ["tsp", "jssp", "vrp"])

    def can_follow(self, other: "OperatorNode") -> bool:
        """Check if this operator can follow another in a sequence."""
        # Construction operators can start a sequence
        if self.category == OperatorCategory.CONSTRUCTION:
            return other is None or other.category != OperatorCategory.CONSTRUCTION

        # Local search needs an initial solution
        if self.category == OperatorCategory.LOCAL_SEARCH:
            return other is not None and (
                other.category == OperatorCategory.CONSTRUCTION
                or "has_solution" in other.effects
            )

        # Perturbation needs an existing solution
        if self.category == OperatorCategory.PERTURBATION:
            return other is not None and "has_solution" in other.effects

        return True


class DataStructureNode(AKGNode):
    """Node representing a data structure used by operators."""

    node_type: NodeType = NodeType.DATA_STRUCTURE
    memory_complexity: str = "O(n)"
    supported_operations: list[str] = Field(default_factory=list)


class PropertyNode(AKGNode):
    """Node representing a problem property that affects operator selection."""

    node_type: NodeType = NodeType.PROPERTY
    value_type: str = "numeric"  # numeric, boolean, categorical
    range: tuple[float, float] | None = None


class AKGEdge(BaseModel):
    """Edge connecting two nodes in the AKG.

    Edges support three levels of LLM knowledge:
    - Level 1 (Topology): source/target define which operators can connect
    - Level 2 (Weights): weight encodes transition preference
    - Level 3 (Conditions): conditions control WHEN the transition should be taken

    Example with condition:
        AKGEdge(
            source="two_opt",
            target="double_bridge",
            edge_type=EdgeType.SEQUENTIAL,
            weight=0.6,
            conditions=[
                EdgeCondition(
                    condition_type=ConditionType.STAGNATION,
                    threshold=3.0,
                    reason="Escape after 3 generations without improvement"
                )
            ],
            condition_boost=1.5  # 50% more likely when condition met
        )
    """

    source: str
    target: str
    edge_type: EdgeType
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Level 3: Conditional transitions
    # Import at runtime to avoid circular imports
    conditions: list[Any] = Field(default_factory=list)  # list[EdgeCondition]
    condition_boost: float = Field(ge=0.0, le=5.0, default=1.0)

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AKGEdge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.edge_type == other.edge_type
        )

    def evaluate_conditions(self, context: "ExecutionContext") -> tuple[bool, float]:
        """Evaluate all conditions against the current execution context.

        Args:
            context: Runtime execution state

        Returns:
            Tuple of (all_conditions_satisfied, effective_boost)
            - all_conditions_satisfied: True if all conditions are met (or no conditions)
            - effective_boost: condition_boost if conditions met, 1.0 otherwise
        """
        if not self.conditions:
            return True, 1.0  # No conditions = always available, no boost

        # All conditions must be satisfied (AND semantics)
        all_met = all(c.evaluate(context) for c in self.conditions)
        boost = self.condition_boost if all_met else 1.0

        return all_met, boost

    def has_conditions(self) -> bool:
        """Check if this edge has any conditions."""
        return len(self.conditions) > 0
