"""Permutation Family Context.

Base class for all permutation-based optimization problems where
solutions are orderings of elements [0, 1, ..., n-1].

Examples: TSP, JSSP, VRP, PFSP, QAP
"""

from __future__ import annotations

from abc import abstractmethod
import random
from typing import Optional

from src.geakg.contexts.base import FamilyContext, OptimizationFamily


class PermutationContext(FamilyContext[list[int]]):
    """Context for permutation-based optimization.

    Solutions are represented as lists of integers [0, n-1] where each
    integer appears exactly once.

    Subclasses (domain contexts like TSPContext) implement the abstract
    methods with domain-specific logic.
    """

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.PERMUTATION

    # =========================================================================
    # Permutation-specific operations
    # =========================================================================

    def swap(self, solution: list[int], i: int, j: int) -> list[int]:
        """Swap elements at positions i and j.

        Args:
            solution: Current permutation
            i: First position
            j: Second position

        Returns:
            New permutation with swapped elements
        """
        result = solution.copy()
        result[i], result[j] = result[j], result[i]
        return result

    def insert(self, solution: list[int], i: int, j: int) -> list[int]:
        """Remove element at i and insert at position j.

        Args:
            solution: Current permutation
            i: Position to remove from
            j: Position to insert at

        Returns:
            New permutation with element relocated
        """
        result = solution.copy()
        elem = result.pop(i)
        result.insert(j, elem)
        return result

    def reverse(self, solution: list[int], i: int, j: int) -> list[int]:
        """Reverse segment from position i to j (inclusive).

        Also known as 2-opt move for TSP.

        Args:
            solution: Current permutation
            i: Start of segment
            j: End of segment (inclusive)

        Returns:
            New permutation with reversed segment
        """
        if i > j:
            i, j = j, i
        result = solution.copy()
        result[i : j + 1] = reversed(result[i : j + 1])
        return result

    def or_opt(self, solution: list[int], i: int, length: int, j: int) -> list[int]:
        """Move a segment of 'length' elements starting at i to position j.

        Or-opt is a generalization of insert for multiple consecutive elements.

        Args:
            solution: Current permutation
            i: Start of segment to move
            length: Number of elements to move (typically 1-3)
            j: Position to insert the segment

        Returns:
            New permutation with segment relocated
        """
        n = len(solution)
        length = min(length, n - i)

        result = solution.copy()
        segment = result[i : i + length]
        del result[i : i + length]

        # Adjust j if it was after the removed segment
        if j > i:
            j -= length

        for k, elem in enumerate(segment):
            result.insert(j + k, elem)

        return result

    def delta_swap(self, solution: list[int], i: int, j: int) -> float:
        """Calculate cost change for swap without applying.

        Override in domain contexts for O(1) computation.

        Args:
            solution: Current permutation
            i: First position
            j: Second position

        Returns:
            Cost change (negative = improvement)
        """
        return self.delta(solution, "swap", i, j)

    def delta_reverse(self, solution: list[int], i: int, j: int) -> float:
        """Calculate cost change for reverse (2-opt) without applying.

        Override in domain contexts for O(1) computation.

        Args:
            solution: Current permutation
            i: Start of segment
            j: End of segment

        Returns:
            Cost change (negative = improvement)
        """
        return self.delta(solution, "reverse", i, j)

    def delta_insert(self, solution: list[int], i: int, j: int) -> float:
        """Calculate cost change for insert without applying.

        Args:
            solution: Current permutation
            i: Position to remove from
            j: Position to insert at

        Returns:
            Cost change (negative = improvement)
        """
        return self.delta(solution, "insert", i, j)

    def apply_move(
        self, solution: list[int], move: str, i: int, j: int
    ) -> list[int] | None:
        """Apply a permutation move.

        Args:
            solution: Current permutation
            move: "swap", "insert", or "reverse"
            i: First position
            j: Second position

        Returns:
            New solution or None if invalid move
        """
        if move == "swap":
            return self.swap(solution, i, j)
        elif move == "insert":
            return self.insert(solution, i, j)
        elif move == "reverse":
            return self.reverse(solution, i, j)
        return None

    # =========================================================================
    # Universal methods (required by OptimizationContext)
    # =========================================================================

    def random_solution(self) -> list[int]:
        """Generate a random valid permutation."""
        perm = list(range(self.dimension))
        random.shuffle(perm)
        return perm

    def copy(self, solution: list[int]) -> list[int]:
        """Deep copy of permutation."""
        return solution.copy()

    def valid(self, solution: list[int]) -> bool:
        """Check if solution is a valid permutation.

        Valid permutation contains each element 0..n-1 exactly once.
        """
        if len(solution) != self.dimension:
            return False
        return set(solution) == set(range(self.dimension))

    # =========================================================================
    # Abstract methods (must be implemented by domain contexts)
    # =========================================================================

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'tsp', 'jssp')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of elements in the permutation."""

    @abstractmethod
    def evaluate(self, solution: list[int]) -> float:
        """Evaluate permutation cost (domain-specific)."""


__all__ = ["PermutationContext"]
