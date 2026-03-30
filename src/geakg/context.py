"""Domain-agnostic context protocol.

Inspired by Butler Lampson's "Hints for Computer System Design":
- "An interface should capture the minimum essentials of an abstraction"
- "Keep secrets of the implementation"
- "Make it fast, rather than general or powerful"

This protocol defines the minimal interface that domain contexts must implement
to enable truly domain-agnostic operators that can transfer across TSP, VRP, JSSP, etc.
"""

from typing import Any, Protocol


class DomainContext(Protocol):
    """Universal interface for domain-specific operations.

    5 methods. No more. No less.

    This is the KEY abstraction enabling domain-agnostic operators.
    All domain-specific knowledge (distance_matrix, demands, capacity, etc.)
    is hidden behind these 5 methods.

    Operators using this interface can be trained on TSP and directly
    executed on VRP without code modification.
    """

    def cost(self, solution: list, i: int) -> float:
        """Cost contribution of element at index i.

        Args:
            solution: Current solution (list representation)
            i: Index of element to evaluate

        Returns:
            Cost contribution of element i to the total solution cost.

        Examples:
            TSP: distance to previous city + distance to next city
            VRP: distance contribution + any capacity penalty
            JSSP: contribution to makespan (critical path)
        """
        ...

    def delta(self, solution: list, move: str, i: int, j: int) -> float:
        """Delta cost if move(i,j) were applied. Does NOT modify solution.

        This enables efficient "what-if" evaluation without executing moves.
        Following Lampson: "Make it fast" - delta should be O(1) when possible.

        Args:
            solution: Current solution
            move: Move type ("swap", "2opt", "insert", etc.)
            i: First index
            j: Second index

        Returns:
            Change in total cost if move were applied (negative = improvement)
        """
        ...

    def neighbors(self, solution: list, i: int, k: int) -> list[int]:
        """K indices most related to element at index i.

        Enables intelligent local search that focuses on promising moves.

        Args:
            solution: Current solution
            i: Index of reference element
            k: Number of neighbors to return

        Returns:
            List of k indices most related to element at i.

        Examples:
            TSP: K nearest cities by distance
            VRP: K nearest customers (same or different route)
            JSSP: K most related operations (same job/machine)
        """
        ...

    def evaluate(self, solution: list) -> float:
        """Total solution cost (fitness).

        Args:
            solution: Solution to evaluate

        Returns:
            Total cost (lower is better for minimization problems)
        """
        ...

    def valid(self, solution: list) -> bool:
        """Check if solution satisfies domain constraints.

        Args:
            solution: Solution to validate

        Returns:
            True if solution is valid for the domain.

        Examples:
            TSP: All cities visited exactly once
            VRP: All customers visited, capacity respected
            JSSP: All operations scheduled, precedence respected
        """
        ...


def verify_context(ctx: Any) -> bool:
    """Verify that an object implements the DomainContext protocol.

    Args:
        ctx: Object to verify

    Returns:
        True if ctx implements all required methods
    """
    required_methods = ["cost", "delta", "neighbors", "evaluate", "valid"]
    return all(hasattr(ctx, method) and callable(getattr(ctx, method))
               for method in required_methods)
