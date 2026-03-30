"""Symbolic constraint validator for LLM proposals.

The validator ensures that operations proposed by the LLM are valid
according to the AKG structure and grammar rules.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import EdgeType, OperatorCategory, OperatorNode


class ProposedOperation(BaseModel):
    """An operation proposed by the LLM."""

    operation_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ValidationResult(BaseModel):
    """Result of validating a proposed operation."""

    valid: bool
    violations: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @property
    def feedback_message(self) -> str:
        """Generate feedback message for LLM."""
        if self.valid:
            return "Operation is valid."

        msg = "Operation rejected:\n"
        for v in self.violations:
            msg += f"  - {v}\n"

        if self.suggestions:
            msg += "\nSuggested alternatives:\n"
            for s in self.suggestions[:3]:
                msg += f"  - {s}\n"

        return msg


class ConstraintValidator:
    """Validates LLM-proposed operations against AKG constraints.

    The validator implements the symbolic constraint engine that ensures
    all proposed operations are valid according to:
    1. Node existence in AKG
    2. Valid transitions from current state
    3. Precondition satisfaction
    4. Grammar rules (construction -> improvement flow)
    """

    def __init__(self, akg: AlgorithmicKnowledgeGraph) -> None:
        """Initialize validator with AKG.

        Args:
            akg: The Algorithmic Knowledge Graph
        """
        self.akg = akg

    def validate(
        self,
        proposed: ProposedOperation,
        current_operators: list[str],
        strict: bool = True,
    ) -> ValidationResult:
        """Validate a proposed operation.

        Args:
            proposed: The proposed operation from LLM
            current_operators: Current algorithm operator sequence
            strict: If True, enforce all constraints

        Returns:
            ValidationResult with validity and feedback
        """
        violations = []
        suggestions = []

        # 1. Check if operation exists in AKG
        node = self.akg.get_node(proposed.operation_id)
        if node is None:
            violations.append(f"Unknown operation: '{proposed.operation_id}'")
            suggestions.extend(self._suggest_similar_operations(proposed.operation_id))
            return ValidationResult(
                valid=False,
                violations=violations,
                suggestions=suggestions,
                confidence=0.0,
            )

        # 2. Check if it's an operator node
        if not isinstance(node, OperatorNode):
            violations.append(f"'{proposed.operation_id}' is not an operator")
            return ValidationResult(
                valid=False,
                violations=violations,
                suggestions=self._get_all_operators(),
                confidence=0.0,
            )

        # 3. Check valid transition from current state
        if current_operators:
            last_op = current_operators[-1]
            valid_next = self.akg.get_valid_transitions(last_op, EdgeType.SEQUENTIAL)

            if proposed.operation_id not in valid_next:
                violations.append(
                    f"Invalid transition: '{last_op}' -> '{proposed.operation_id}'"
                )
                suggestions.extend(valid_next[:5])
        else:
            # First operation must be construction
            if node.category != OperatorCategory.CONSTRUCTION:
                violations.append(
                    f"First operation must be a construction operator, "
                    f"got '{node.category.value}'"
                )
                construction_ops = self.akg.get_operators_by_category(
                    OperatorCategory.CONSTRUCTION
                )
                suggestions.extend([op.id for op in construction_ops[:5]])

        # 4. Check preconditions
        precondition_violations = self._check_preconditions(node, current_operators)
        violations.extend(precondition_violations)

        # 5. Check grammar rules
        grammar_violations = self._check_grammar(node, current_operators)
        if strict:
            violations.extend(grammar_violations)

        # Calculate confidence based on edge weight
        confidence = 1.0
        if current_operators and not violations:
            last_op = current_operators[-1]
            edge = self.akg.edges.get((last_op, proposed.operation_id))
            if edge:
                confidence = edge.weight

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations,
            suggestions=suggestions if violations else [],
            confidence=confidence,
        )

    def _check_preconditions(
        self, node: OperatorNode, current_operators: list[str]
    ) -> list[str]:
        """Check if preconditions for an operator are satisfied.

        Args:
            node: The operator node to check
            current_operators: Current operator sequence

        Returns:
            List of precondition violations
        """
        violations = []

        # Collect effects from current operators
        current_effects = set()
        for op_id in current_operators:
            op_node = self.akg.get_node(op_id)
            if isinstance(op_node, OperatorNode):
                current_effects.update(op_node.effects)

        # Check each precondition
        for precond in node.preconditions:
            if precond not in current_effects:
                # Some preconditions are about problem features, not operator effects
                if precond not in ("has_coordinates", "has_pheromones"):
                    violations.append(
                        f"Precondition not satisfied: '{precond}' required for '{node.id}'"
                    )

        return violations

    def _check_grammar(
        self, node: OperatorNode, current_operators: list[str]
    ) -> list[str]:
        """Check grammar rules for algorithm structure.

        Grammar rules:
        - Algorithms should start with construction
        - Multiple constructions in sequence are discouraged
        - Perturbation should follow local search
        - Meta-heuristics can loop

        Args:
            node: The operator to add
            current_operators: Current operator sequence

        Returns:
            List of grammar violations
        """
        violations = []

        if not current_operators:
            return violations

        # Count operators by category
        category_counts = {cat: 0 for cat in OperatorCategory}
        for op_id in current_operators:
            op_node = self.akg.get_node(op_id)
            if isinstance(op_node, OperatorNode):
                category_counts[op_node.category] += 1

        # Rule: Don't add more than 2 construction operators
        if node.category == OperatorCategory.CONSTRUCTION:
            if category_counts[OperatorCategory.CONSTRUCTION] >= 2:
                violations.append(
                    "Too many construction operators (max 2)"
                )

        # Rule: Perturbation should have improvement before it
        if node.category == OperatorCategory.PERTURBATION:
            if category_counts[OperatorCategory.LOCAL_SEARCH] == 0:
                violations.append(
                    "Perturbation should follow local search improvement"
                )

        return violations

    def _suggest_similar_operations(self, unknown_op: str) -> list[str]:
        """Suggest similar operation names for typos/hallucinations.

        Args:
            unknown_op: The unknown operation ID

        Returns:
            List of similar valid operation IDs
        """
        all_ops = self._get_all_operators()

        # Simple similarity: common prefix or substring
        suggestions = []
        unknown_lower = unknown_op.lower()

        for op in all_ops:
            op_lower = op.lower()
            # Check prefix match
            if op_lower.startswith(unknown_lower[:3]) or unknown_lower.startswith(op_lower[:3]):
                suggestions.append(op)
            # Check substring match
            elif unknown_lower in op_lower or op_lower in unknown_lower:
                suggestions.append(op)

        return suggestions[:5] if suggestions else all_ops[:5]

    def _get_all_operators(self) -> list[str]:
        """Get all operator IDs from AKG.

        Returns:
            List of operator IDs
        """
        return [op.id for op in self.akg.get_operator_nodes()]

    def get_valid_operations(
        self, current_operators: list[str]
    ) -> list[tuple[str, float]]:
        """Get all valid next operations with their weights.

        Args:
            current_operators: Current operator sequence

        Returns:
            List of (operation_id, weight) tuples, sorted by weight
        """
        if not current_operators:
            # Return construction operators
            construction_ops = self.akg.get_operators_by_category(
                OperatorCategory.CONSTRUCTION
            )
            return [(op.id, 1.0) for op in construction_ops]

        last_op = current_operators[-1]
        valid_next = self.akg.get_valid_transitions(last_op)

        # Get weights
        result = []
        for op_id in valid_next:
            edge = self.akg.edges.get((last_op, op_id))
            weight = edge.weight if edge else 0.5
            result.append((op_id, weight))

        # Sort by weight descending
        result.sort(key=lambda x: x[1], reverse=True)
        return result
