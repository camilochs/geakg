"""Mask generator for valid operations.

Generates masks that indicate which operations are valid given the
current algorithm state. These masks are used to constrain LLM proposals.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import EdgeType, OperatorCategory, OperatorNode


class OperationMask(BaseModel):
    """Mask indicating valid operations.

    The mask maps each operator ID to a validity score between 0 and 1.
    - 0.0: Operation is invalid (hard constraint violation)
    - 0.0-0.5: Operation is discouraged (soft constraint)
    - 0.5-1.0: Operation is valid with varying preference
    - 1.0: Operation is strongly recommended
    """

    operator_validity: dict[str, float] = Field(default_factory=dict)
    current_state: list[str] = Field(default_factory=list)
    explanation: str = ""

    @property
    def valid_operators(self) -> list[str]:
        """Get list of valid operator IDs (validity > 0)."""
        return [op for op, v in self.operator_validity.items() if v > 0]

    @property
    def ranked_operators(self) -> list[tuple[str, float]]:
        """Get operators ranked by validity score."""
        sorted_ops = sorted(
            self.operator_validity.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(op, v) for op, v in sorted_ops if v > 0]

    def to_prompt_format(self, top_k: int = 10) -> str:
        """Format mask for inclusion in LLM prompt.

        Args:
            top_k: Maximum number of operations to include

        Returns:
            Formatted string for prompt
        """
        ranked = self.ranked_operators[:top_k]
        if not ranked:
            return "No valid operations available."

        lines = ["Valid operations (sorted by preference):"]
        for op, score in ranked:
            level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
            lines.append(f"  - {op} ({level})")

        return "\n".join(lines)


class MaskGenerator:
    """Generates validity masks for operations based on AKG constraints.

    The mask generator produces soft constraints that guide the LLM toward
    valid operations while still allowing exploration of the graph.
    """

    def __init__(self, akg: AlgorithmicKnowledgeGraph) -> None:
        """Initialize mask generator.

        Args:
            akg: The Algorithmic Knowledge Graph
        """
        self.akg = akg

    def generate_mask(
        self,
        current_operators: list[str],
        problem_context: dict[str, Any] | None = None,
    ) -> OperationMask:
        """Generate validity mask for current state.

        Args:
            current_operators: Current algorithm operator sequence
            problem_context: Optional problem-specific context

        Returns:
            OperationMask with validity scores
        """
        validity = {}
        all_operators = self.akg.get_operator_nodes()

        if not current_operators:
            # Starting state: only construction operators are valid
            for op in all_operators:
                if op.category == OperatorCategory.CONSTRUCTION:
                    validity[op.id] = 1.0
                else:
                    validity[op.id] = 0.0

            return OperationMask(
                operator_validity=validity,
                current_state=current_operators,
                explanation="Starting state: construction operators only",
            )

        # Get valid transitions from last operator
        last_op = current_operators[-1]
        valid_next = set(self.akg.get_valid_transitions(last_op, EdgeType.SEQUENTIAL))

        # Assign validity scores
        for op in all_operators:
            if op.id in valid_next:
                # Valid transition: use edge weight
                edge = self.akg.edges.get((last_op, op.id))
                base_score = edge.weight if edge else 0.5

                # Apply modifiers
                score = self._apply_modifiers(
                    op, current_operators, base_score, problem_context
                )
                validity[op.id] = score
            else:
                # Invalid transition
                validity[op.id] = 0.0

        explanation = self._generate_explanation(current_operators, validity)

        return OperationMask(
            operator_validity=validity,
            current_state=current_operators.copy(),
            explanation=explanation,
        )

    def _apply_modifiers(
        self,
        operator: OperatorNode,
        current_operators: list[str],
        base_score: float,
        problem_context: dict[str, Any] | None,
    ) -> float:
        """Apply score modifiers based on context.

        Args:
            operator: The operator to score
            current_operators: Current operator sequence
            base_score: Base validity score
            problem_context: Optional problem context

        Returns:
            Modified validity score
        """
        score = base_score

        # Count operator categories in current algorithm
        category_counts = self._count_categories(current_operators)

        # Penalize repetition of same category
        if category_counts.get(operator.category, 0) >= 2:
            score *= 0.7

        # Boost local search after construction
        if operator.category == OperatorCategory.LOCAL_SEARCH:
            if category_counts.get(OperatorCategory.CONSTRUCTION, 0) > 0:
                if category_counts.get(OperatorCategory.LOCAL_SEARCH, 0) == 0:
                    score *= 1.2

        # Boost perturbation after local search has been applied
        if operator.category == OperatorCategory.PERTURBATION:
            if category_counts.get(OperatorCategory.LOCAL_SEARCH, 0) > 0:
                score *= 1.1

        # Apply problem-specific modifiers if provided
        if problem_context:
            score = self._apply_problem_modifiers(operator, score, problem_context)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _apply_problem_modifiers(
        self,
        operator: OperatorNode,
        score: float,
        context: dict[str, Any],
    ) -> float:
        """Apply problem-specific score modifiers.

        Args:
            operator: The operator
            score: Current score
            context: Problem context

        Returns:
            Modified score
        """
        # Boost operators that work well for problem size
        problem_size = context.get("dimension", 100)

        if problem_size < 50:
            # Small problems: boost exhaustive local search
            if operator.id in ("two_opt", "three_opt", "lin_kernighan"):
                score *= 1.15
        elif problem_size > 500:
            # Large problems: boost faster heuristics
            if operator.id in ("greedy_nearest_neighbor", "two_opt"):
                score *= 1.1
            # Penalize expensive operators
            if operator.id in ("three_opt", "lin_kernighan"):
                score *= 0.8

        # Check if operator supports the problem domain
        domain = context.get("domain", "tsp")
        if hasattr(operator, "domains") and domain not in operator.domains:
            score *= 0.5

        return score

    def _count_categories(
        self, operators: list[str]
    ) -> dict[OperatorCategory, int]:
        """Count operators by category.

        Args:
            operators: List of operator IDs

        Returns:
            Dict mapping categories to counts
        """
        counts: dict[OperatorCategory, int] = {}

        for op_id in operators:
            node = self.akg.get_node(op_id)
            if isinstance(node, OperatorNode):
                counts[node.category] = counts.get(node.category, 0) + 1

        return counts

    def _generate_explanation(
        self,
        current_operators: list[str],
        validity: dict[str, float],
    ) -> str:
        """Generate human-readable explanation of mask.

        Args:
            current_operators: Current operator sequence
            validity: Validity scores

        Returns:
            Explanation string
        """
        n_valid = sum(1 for v in validity.values() if v > 0)
        n_high = sum(1 for v in validity.values() if v >= 0.7)

        if not current_operators:
            return f"Starting state: {n_valid} construction operators available"

        last_op = current_operators[-1]
        return (
            f"After '{last_op}': {n_valid} valid transitions, "
            f"{n_high} highly recommended"
        )

    def get_top_k_operations(
        self,
        current_operators: list[str],
        k: int = 5,
        problem_context: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Get top K valid operations.

        Args:
            current_operators: Current operator sequence
            k: Number of operations to return
            problem_context: Optional problem context

        Returns:
            List of (operator_id, score) tuples
        """
        mask = self.generate_mask(current_operators, problem_context)
        return mask.ranked_operators[:k]
