"""Universal Optimization Context Interface.

Defines the base abstraction for all optimization problems, following
the principle of minimal interfaces (Lampson's "Keep Secrets").

Architecture:
    OptimizationContext (Universal)
           ↓
    FamilyContext (Permutation, Binary, Continuous, Partition)
           ↓
    DomainContext (TSP, Knapsack, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar, Optional

T = TypeVar("T")  # Solution type (varies by family)


class OptimizationFamily(str, Enum):
    """Families of problems based on solution representation."""

    PERMUTATION = "permutation"
    """Solutions are orderings of elements. Examples: TSP, JSSP, VRP, PFSP."""

    BINARY = "binary"
    """Solutions are bit vectors. Examples: Knapsack, Set Cover, Feature Selection."""

    CONTINUOUS = "continuous"
    """Solutions are real-valued vectors. Examples: Function Optimization."""

    PARTITION = "partition"
    """Solutions assign items to groups. Examples: Bin Packing, Graph Partitioning."""

    ARCHITECTURE = "architecture"
    """Solutions are neural architecture DAGs. Examples: NAS, AutoML."""


class OptimizationContext(ABC, Generic[T]):
    """Universal interface for all optimization problems.

    This is the root of the context hierarchy. All domain-specific
    contexts must implement these core methods.

    The interface is intentionally minimal to enable maximum portability
    of operators across domains.

    Type Parameters:
        T: The solution type (list[int] for permutation/binary,
           list[float] for continuous, etc.)
    """

    @property
    @abstractmethod
    def family(self) -> OptimizationFamily:
        """Get the optimization family.

        Returns:
            One of: PERMUTATION, BINARY, CONTINUOUS, PARTITION
        """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get the domain identifier.

        Returns:
            Domain name (e.g., 'tsp', 'knapsack', 'function_opt')
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the problem dimension.

        For permutation: number of elements to order
        For binary: number of bits
        For continuous: number of variables
        For partition: number of items to assign

        Returns:
            Problem size/dimension
        """

    @abstractmethod
    def evaluate(self, solution: T) -> float:
        """Evaluate solution quality (cost).

        IMPORTANT: Lower values are always better (minimization).
        For maximization problems, return -value.

        Args:
            solution: A complete solution

        Returns:
            Cost value (lower is better)
        """

    @abstractmethod
    def valid(self, solution: T) -> bool:
        """Check if solution satisfies all constraints.

        Args:
            solution: Solution to validate

        Returns:
            True if solution is feasible
        """

    @abstractmethod
    def random_solution(self) -> T:
        """Generate a random valid solution.

        Used for initialization and diversification.

        Returns:
            A valid random solution
        """

    @abstractmethod
    def copy(self, solution: T) -> T:
        """Create a deep copy of the solution.

        Args:
            solution: Solution to copy

        Returns:
            Independent copy of the solution
        """

    # =========================================================================
    # Optional methods with default implementations
    # =========================================================================

    def cost(self, solution: T, i: int) -> float:
        """Cost contribution of element at position i.

        Optional method - provides more granular cost information.
        Default implementation returns 0 (override for efficiency).

        Args:
            solution: Current solution
            i: Position index

        Returns:
            Local cost contribution at position i
        """
        return 0.0

    def delta(self, solution: T, move: str, i: int, j: int) -> float:
        """Calculate cost change for a move without applying it.

        Optional but highly recommended for efficiency.
        Default uses full evaluation (slow).

        Args:
            solution: Current solution
            move: Move type (family-specific, e.g., "swap", "flip")
            i: First position
            j: Second position

        Returns:
            Change in cost (negative = improvement)
        """
        # Fallback: compute actual delta (inefficient)
        original = self.evaluate(solution)
        modified = self.apply_move(solution, move, i, j)
        if modified is None:
            return float("inf")
        return self.evaluate(modified) - original

    def apply_move(self, solution: T, move: str, i: int, j: int) -> T | None:
        """Apply a move and return the new solution.

        Override in family contexts for specific move types.

        Args:
            solution: Current solution
            move: Move type
            i: First position
            j: Second position

        Returns:
            New solution or None if invalid
        """
        return None

    @property
    def instance_data(self) -> dict:
        """Get instance-specific data for operators that need it.

        This is a controlled escape hatch for operators that need
        direct access to problem data. Use sparingly.

        Returns:
            Dictionary with instance parameters
        """
        return {"dimension": self.dimension}


class FamilyContext(OptimizationContext[T], ABC):
    """Base class for family-specific contexts.

    Adds family-specific methods that are common across all domains
    within that family (e.g., swap for permutation, flip for binary).
    """

    pass


__all__ = [
    "OptimizationFamily",
    "OptimizationContext",
    "FamilyContext",
]
