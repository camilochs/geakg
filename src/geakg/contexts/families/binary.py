"""Binary Family Context.

Base class for all binary-vector optimization problems where
solutions are bit vectors [0, 1, 0, 1, ...].

Examples: Knapsack, Set Cover, Feature Selection, MAX-SAT
"""

from __future__ import annotations

from abc import abstractmethod
import random
from typing import Optional

from src.geakg.contexts.base import FamilyContext, OptimizationFamily


class BinaryContext(FamilyContext[list[int]]):
    """Context for binary-vector optimization.

    Solutions are represented as lists of 0s and 1s.
    Typically used for selection problems (select or not).

    Subclasses implement domain-specific evaluation and constraints.
    """

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.BINARY

    # =========================================================================
    # Binary-specific operations
    # =========================================================================

    def flip(self, solution: list[int], i: int) -> list[int]:
        """Flip bit at position i.

        Args:
            solution: Current bit vector
            i: Position to flip

        Returns:
            New solution with flipped bit
        """
        result = solution.copy()
        result[i] = 1 - result[i]
        return result

    def flip_multiple(self, solution: list[int], indices: list[int]) -> list[int]:
        """Flip multiple bits at once.

        Args:
            solution: Current bit vector
            indices: Positions to flip

        Returns:
            New solution with flipped bits
        """
        result = solution.copy()
        for i in indices:
            result[i] = 1 - result[i]
        return result

    def set_bit(self, solution: list[int], i: int, value: int) -> list[int]:
        """Set bit at position i to specific value.

        Args:
            solution: Current bit vector
            i: Position to set
            value: 0 or 1

        Returns:
            New solution with set bit
        """
        result = solution.copy()
        result[i] = value
        return result

    def swap_values(self, solution: list[int], i: int, j: int) -> list[int]:
        """Swap a 0 and a 1 (find a 0 at i-th zero, 1 at j-th one).

        More efficient than random flipping for balanced changes.

        Args:
            solution: Current bit vector
            i: Index to swap (0 to 1)
            j: Index to swap (1 to 0)

        Returns:
            New solution with swapped bits
        """
        result = solution.copy()
        result[i], result[j] = result[j], result[i]
        return result

    def delta_flip(self, solution: list[int], i: int) -> float:
        """Calculate cost change for flipping bit i without applying.

        Override in domain contexts for O(1) computation.

        Args:
            solution: Current bit vector
            i: Position to flip

        Returns:
            Cost change (negative = improvement)
        """
        return self.delta(solution, "flip", i, i)

    def apply_move(
        self, solution: list[int], move: str, i: int, j: int
    ) -> list[int] | None:
        """Apply a binary move.

        Args:
            solution: Current bit vector
            move: "flip" or "swap"
            i: Position (for flip) or first position (for swap)
            j: Ignored for flip, second position for swap

        Returns:
            New solution or None if invalid
        """
        if move == "flip":
            return self.flip(solution, i)
        elif move == "swap":
            return self.swap_values(solution, i, j)
        return None

    # =========================================================================
    # Utility methods for binary problems
    # =========================================================================

    def count_ones(self, solution: list[int]) -> int:
        """Count number of selected items (1s).

        Args:
            solution: Bit vector

        Returns:
            Number of 1s in solution
        """
        return sum(solution)

    def count_zeros(self, solution: list[int]) -> int:
        """Count number of unselected items (0s).

        Args:
            solution: Bit vector

        Returns:
            Number of 0s in solution
        """
        return len(solution) - sum(solution)

    def get_selected_indices(self, solution: list[int]) -> list[int]:
        """Get indices of all selected items (1s).

        Args:
            solution: Bit vector

        Returns:
            List of indices where solution[i] == 1
        """
        return [i for i, v in enumerate(solution) if v == 1]

    def get_unselected_indices(self, solution: list[int]) -> list[int]:
        """Get indices of all unselected items (0s).

        Args:
            solution: Bit vector

        Returns:
            List of indices where solution[i] == 0
        """
        return [i for i, v in enumerate(solution) if v == 0]

    def hamming_distance(self, sol1: list[int], sol2: list[int]) -> int:
        """Calculate Hamming distance between two solutions.

        Args:
            sol1: First bit vector
            sol2: Second bit vector

        Returns:
            Number of positions where they differ
        """
        return sum(a != b for a, b in zip(sol1, sol2))

    # =========================================================================
    # Constraint handling (common in binary problems)
    # =========================================================================

    def repair_greedy(self, solution: list[int]) -> list[int]:
        """Repair infeasible solution using greedy strategy.

        Override in domain contexts for domain-specific repair.
        Default: no repair (returns as-is if valid, empty otherwise).

        Args:
            solution: Potentially infeasible solution

        Returns:
            Feasible solution
        """
        if self.valid(solution):
            return solution
        # Default: return empty solution (all zeros)
        return [0] * self.dimension

    def repair_random(self, solution: list[int]) -> list[int]:
        """Repair by randomly removing items until feasible.

        Override for domain-specific random repair.

        Args:
            solution: Potentially infeasible solution

        Returns:
            Feasible solution
        """
        result = solution.copy()
        selected = self.get_selected_indices(result)
        random.shuffle(selected)

        while not self.valid(result) and selected:
            idx = selected.pop()
            result[idx] = 0

        return result

    # =========================================================================
    # Universal methods (required by OptimizationContext)
    # =========================================================================

    def random_solution(self) -> list[int]:
        """Generate a random valid binary solution.

        Default: random bits with 50% probability.
        Override for constrained problems.
        """
        solution = [random.randint(0, 1) for _ in range(self.dimension)]
        # Try to repair if invalid
        if not self.valid(solution):
            solution = self.repair_greedy(solution)
        return solution

    def copy(self, solution: list[int]) -> list[int]:
        """Deep copy of bit vector."""
        return solution.copy()

    def valid(self, solution: list[int]) -> bool:
        """Check if solution is valid binary vector.

        Override in domain contexts for constraint checking.
        Default: checks length and values are 0 or 1.
        """
        if len(solution) != self.dimension:
            return False
        return all(v in (0, 1) for v in solution)

    # =========================================================================
    # Abstract methods (must be implemented by domain contexts)
    # =========================================================================

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'knapsack', 'set_cover')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of bits (items to select from)."""

    @abstractmethod
    def evaluate(self, solution: list[int]) -> float:
        """Evaluate solution cost (domain-specific)."""


__all__ = ["BinaryContext"]
