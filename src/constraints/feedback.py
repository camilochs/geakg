"""Feedback generator for LLM constraint violations.

Generates structured feedback when the LLM proposes invalid operations,
helping it learn from rejections and converge toward valid proposals.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorCategory
from src.constraints.validator import ValidationResult


class FeedbackMessage(BaseModel):
    """Structured feedback message for LLM."""

    rejection_reason: str
    violated_constraints: list[str] = Field(default_factory=list)
    suggested_alternatives: list[str] = Field(default_factory=list)
    hint: str = ""
    severity: str = "error"  # error, warning, info

    def to_prompt(self) -> str:
        """Format feedback for inclusion in LLM prompt.

        Returns:
            Formatted feedback string
        """
        lines = [f"[{self.severity.upper()}] {self.rejection_reason}"]

        if self.violated_constraints:
            lines.append("\nViolated constraints:")
            for c in self.violated_constraints:
                lines.append(f"  - {c}")

        if self.suggested_alternatives:
            lines.append("\nValid alternatives:")
            for alt in self.suggested_alternatives[:5]:
                lines.append(f"  - {alt}")

        if self.hint:
            lines.append(f"\nHint: {self.hint}")

        return "\n".join(lines)


class RejectionLog(BaseModel):
    """Log of rejected proposals for analysis."""

    proposal_id: str
    proposed_operation: str
    current_state: list[str]
    violations: list[str]
    feedback_given: str
    attempt_number: int


class FeedbackGenerator:
    """Generates helpful feedback for LLM when proposals are rejected.

    The feedback generator translates constraint violations into actionable
    guidance that helps the LLM understand why its proposal was rejected
    and what valid alternatives exist.
    """

    def __init__(self, akg: AlgorithmicKnowledgeGraph) -> None:
        """Initialize feedback generator.

        Args:
            akg: The Algorithmic Knowledge Graph
        """
        self.akg = akg
        self.rejection_history: list[RejectionLog] = []
        self._attempt_counter = 0

    def generate_feedback(
        self,
        proposed_operation: str,
        validation_result: ValidationResult,
        current_operators: list[str],
    ) -> FeedbackMessage:
        """Generate feedback for a rejected proposal.

        Args:
            proposed_operation: The rejected operation ID
            validation_result: Result from validator
            current_operators: Current operator sequence

        Returns:
            Structured feedback message
        """
        if validation_result.valid:
            return FeedbackMessage(
                rejection_reason="Operation accepted",
                severity="info",
            )

        # Determine the type of violation
        violation_types = self._categorize_violations(validation_result.violations)

        # Generate appropriate feedback based on violation type
        if "unknown_operation" in violation_types:
            feedback = self._feedback_unknown_operation(
                proposed_operation, validation_result.suggestions
            )
        elif "invalid_transition" in violation_types:
            feedback = self._feedback_invalid_transition(
                proposed_operation, current_operators, validation_result.suggestions
            )
        elif "wrong_category" in violation_types:
            feedback = self._feedback_wrong_category(
                proposed_operation, current_operators
            )
        elif "precondition" in violation_types:
            feedback = self._feedback_precondition(
                proposed_operation, validation_result.violations
            )
        else:
            feedback = self._feedback_generic(validation_result)

        # Log the rejection for analysis
        self._log_rejection(
            proposed_operation,
            current_operators,
            validation_result.violations,
            feedback.to_prompt(),
        )

        return feedback

    def _categorize_violations(self, violations: list[str]) -> set[str]:
        """Categorize violations by type.

        Args:
            violations: List of violation messages

        Returns:
            Set of violation type identifiers
        """
        types = set()

        for v in violations:
            v_lower = v.lower()
            if "unknown" in v_lower:
                types.add("unknown_operation")
            elif "invalid transition" in v_lower:
                types.add("invalid_transition")
            elif "first operation" in v_lower or "construction" in v_lower:
                types.add("wrong_category")
            elif "precondition" in v_lower:
                types.add("precondition")
            elif "too many" in v_lower:
                types.add("limit_exceeded")

        return types if types else {"generic"}

    def _feedback_unknown_operation(
        self, operation: str, suggestions: list[str]
    ) -> FeedbackMessage:
        """Generate feedback for unknown operation.

        Args:
            operation: The unknown operation
            suggestions: Suggested alternatives

        Returns:
            Feedback message
        """
        return FeedbackMessage(
            rejection_reason=f"'{operation}' is not a known operation in the AKG",
            violated_constraints=[f"Operation must exist in the knowledge graph"],
            suggested_alternatives=suggestions,
            hint="Choose from the list of valid operators. Check spelling carefully.",
            severity="error",
        )

    def _feedback_invalid_transition(
        self,
        operation: str,
        current_operators: list[str],
        suggestions: list[str],
    ) -> FeedbackMessage:
        """Generate feedback for invalid transition.

        Args:
            operation: The proposed operation
            current_operators: Current sequence
            suggestions: Valid alternatives

        Returns:
            Feedback message
        """
        last_op = current_operators[-1] if current_operators else "START"

        return FeedbackMessage(
            rejection_reason=f"Cannot transition from '{last_op}' to '{operation}'",
            violated_constraints=[
                f"No valid edge from '{last_op}' to '{operation}' in AKG"
            ],
            suggested_alternatives=suggestions,
            hint=f"After '{last_op}', consider improvement or perturbation operators",
            severity="error",
        )

    def _feedback_wrong_category(
        self, operation: str, current_operators: list[str]
    ) -> FeedbackMessage:
        """Generate feedback for wrong operator category.

        Args:
            operation: The proposed operation
            current_operators: Current sequence

        Returns:
            Feedback message
        """
        # Get construction operators as suggestions
        construction_ops = self.akg.get_operators_by_category(
            OperatorCategory.CONSTRUCTION
        )
        suggestions = [op.id for op in construction_ops[:5]]

        if not current_operators:
            return FeedbackMessage(
                rejection_reason="First operation must be a construction operator",
                violated_constraints=[
                    "Algorithms must start with a construction phase"
                ],
                suggested_alternatives=suggestions,
                hint="Start with greedy_nearest_neighbor or another construction operator",
                severity="error",
            )

        return FeedbackMessage(
            rejection_reason=f"'{operation}' is not appropriate at this stage",
            violated_constraints=["Operator category mismatch"],
            suggested_alternatives=suggestions,
            severity="warning",
        )

    def _feedback_precondition(
        self, operation: str, violations: list[str]
    ) -> FeedbackMessage:
        """Generate feedback for precondition violation.

        Args:
            operation: The proposed operation
            violations: List of violations

        Returns:
            Feedback message
        """
        precond_violations = [v for v in violations if "precondition" in v.lower()]

        return FeedbackMessage(
            rejection_reason=f"Preconditions not met for '{operation}'",
            violated_constraints=precond_violations,
            hint="Ensure required operators have been applied first",
            severity="warning",
        )

    def _feedback_generic(self, result: ValidationResult) -> FeedbackMessage:
        """Generate generic feedback.

        Args:
            result: Validation result

        Returns:
            Feedback message
        """
        return FeedbackMessage(
            rejection_reason="Operation rejected due to constraint violations",
            violated_constraints=result.violations,
            suggested_alternatives=result.suggestions,
            severity="error",
        )

    def _log_rejection(
        self,
        operation: str,
        current_operators: list[str],
        violations: list[str],
        feedback: str,
    ) -> None:
        """Log a rejection for later analysis.

        Args:
            operation: The rejected operation
            current_operators: Current state
            violations: List of violations
            feedback: Feedback given
        """
        self._attempt_counter += 1
        log = RejectionLog(
            proposal_id=f"rej_{self._attempt_counter:04d}",
            proposed_operation=operation,
            current_state=current_operators.copy(),
            violations=violations,
            feedback_given=feedback,
            attempt_number=self._attempt_counter,
        )
        self.rejection_history.append(log)

    def get_rejection_summary(self) -> dict[str, Any]:
        """Get summary of rejection patterns.

        Returns:
            Summary dict with rejection statistics
        """
        if not self.rejection_history:
            return {"total_rejections": 0}

        # Count violations by type
        violation_counts: dict[str, int] = {}
        for log in self.rejection_history:
            for v in log.violations:
                types = self._categorize_violations([v])
                for t in types:
                    violation_counts[t] = violation_counts.get(t, 0) + 1

        # Most rejected operations
        op_counts: dict[str, int] = {}
        for log in self.rejection_history:
            op = log.proposed_operation
            op_counts[op] = op_counts.get(op, 0) + 1

        most_rejected = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_rejections": len(self.rejection_history),
            "violation_types": violation_counts,
            "most_rejected_operations": most_rejected,
        }

    def clear_history(self) -> None:
        """Clear rejection history."""
        self.rejection_history.clear()
        self._attempt_counter = 0
