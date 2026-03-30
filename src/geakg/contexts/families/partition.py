"""Partition Family Context.

Base class for all partition-based optimization problems where
solutions assign items to groups/bins.

Examples: Bin Packing, Graph Partitioning, Clustering, Vehicle Routing
"""

from __future__ import annotations

from abc import abstractmethod
import random
from collections import defaultdict
from typing import Optional

from src.geakg.contexts.base import FamilyContext, OptimizationFamily


class PartitionContext(FamilyContext[list[int]]):
    """Context for partition-based optimization.

    Solutions are represented as lists where solution[i] = group_id
    indicating which group item i belongs to.

    Example: [0, 1, 0, 2, 1] means items 0,2 are in group 0,
    items 1,4 are in group 1, and item 3 is in group 2.

    Subclasses implement domain-specific evaluation and constraints.
    """

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.PARTITION

    # =========================================================================
    # Partition-specific properties
    # =========================================================================

    @property
    @abstractmethod
    def n_groups(self) -> int:
        """Maximum number of groups allowed.

        For bin packing: max bins (often = n_items)
        For k-way partitioning: fixed k
        """

    @property
    def n_items(self) -> int:
        """Number of items to partition (alias for dimension)."""
        return self.dimension

    # =========================================================================
    # Partition-specific operations
    # =========================================================================

    def move(self, solution: list[int], item: int, to_group: int) -> list[int]:
        """Move an item to a different group.

        Args:
            solution: Current partition
            item: Item index to move
            to_group: Target group

        Returns:
            New partition with item moved
        """
        result = solution.copy()
        result[item] = to_group
        return result

    def swap_items(self, solution: list[int], i: int, j: int) -> list[int]:
        """Swap group assignments of two items.

        Args:
            solution: Current partition
            i: First item index
            j: Second item index

        Returns:
            New partition with swapped assignments
        """
        result = solution.copy()
        result[i], result[j] = result[j], result[i]
        return result

    def merge_groups(self, solution: list[int], g1: int, g2: int) -> list[int]:
        """Merge group g2 into group g1.

        Args:
            solution: Current partition
            g1: Target group
            g2: Group to merge (all items move to g1)

        Returns:
            New partition with merged groups
        """
        result = solution.copy()
        for i in range(len(result)):
            if result[i] == g2:
                result[i] = g1
        return result

    def split_group(
        self, solution: list[int], group: int, new_group: int
    ) -> list[int]:
        """Split a group, moving half the items to a new group.

        Args:
            solution: Current partition
            group: Group to split
            new_group: New group for half the items

        Returns:
            New partition with split group
        """
        result = solution.copy()
        items_in_group = [i for i, g in enumerate(result) if g == group]

        if len(items_in_group) < 2:
            return result

        # Move half to new group
        random.shuffle(items_in_group)
        for item in items_in_group[: len(items_in_group) // 2]:
            result[item] = new_group

        return result

    def delta_move(self, solution: list[int], item: int, to_group: int) -> float:
        """Calculate cost change for moving an item.

        Override in domain contexts for O(1) computation.

        Args:
            solution: Current partition
            item: Item to move
            to_group: Target group

        Returns:
            Cost change (negative = improvement)
        """
        return self.delta(solution, "move", item, to_group)

    def apply_move(
        self, solution: list[int], move: str, i: int, j: int
    ) -> list[int] | None:
        """Apply a partition move.

        Args:
            solution: Current partition
            move: "move" (item i to group j) or "swap" (items i and j)
            i: First parameter (item for move, first item for swap)
            j: Second parameter (group for move, second item for swap)

        Returns:
            New solution or None if invalid
        """
        if move == "move":
            return self.move(solution, i, j)
        elif move == "swap":
            return self.swap_items(solution, i, j)
        elif move == "merge":
            return self.merge_groups(solution, i, j)
        return None

    # =========================================================================
    # Group analysis utilities
    # =========================================================================

    def get_groups(self, solution: list[int]) -> dict[int, list[int]]:
        """Get items in each group.

        Args:
            solution: Partition

        Returns:
            Dict mapping group_id -> list of item indices
        """
        groups: dict[int, list[int]] = defaultdict(list)
        for item, group in enumerate(solution):
            groups[group].append(item)
        return dict(groups)

    def group_sizes(self, solution: list[int]) -> dict[int, int]:
        """Get size of each group.

        Args:
            solution: Partition

        Returns:
            Dict mapping group_id -> number of items
        """
        sizes: dict[int, int] = defaultdict(int)
        for group in solution:
            sizes[group] += 1
        return dict(sizes)

    def num_active_groups(self, solution: list[int]) -> int:
        """Count number of non-empty groups.

        Args:
            solution: Partition

        Returns:
            Number of groups with at least one item
        """
        return len(set(solution))

    def group_load(self, solution: list[int], group: int) -> float:
        """Calculate total load/weight of a group.

        Override in domain contexts with actual weights.
        Default: returns number of items.

        Args:
            solution: Partition
            group: Group to measure

        Returns:
            Total load of the group
        """
        return sum(1 for g in solution if g == group)

    def balance_metric(self, solution: list[int]) -> float:
        """Calculate balance metric across groups.

        Higher values indicate more imbalance.
        Override for domain-specific balance measures.

        Args:
            solution: Partition

        Returns:
            Imbalance measure (0 = perfectly balanced)
        """
        sizes = list(self.group_sizes(solution).values())
        if not sizes:
            return 0.0
        avg = sum(sizes) / len(sizes)
        return sum(abs(s - avg) for s in sizes) / len(sizes)

    def largest_group(self, solution: list[int]) -> int:
        """Find the group with most items.

        Args:
            solution: Partition

        Returns:
            Group ID of largest group
        """
        sizes = self.group_sizes(solution)
        return max(sizes.keys(), key=lambda g: sizes[g])

    def smallest_group(self, solution: list[int]) -> int:
        """Find the group with fewest items.

        Args:
            solution: Partition

        Returns:
            Group ID of smallest group
        """
        sizes = self.group_sizes(solution)
        return min(sizes.keys(), key=lambda g: sizes[g])

    # =========================================================================
    # Repair methods
    # =========================================================================

    def compact_groups(self, solution: list[int]) -> list[int]:
        """Renumber groups to be consecutive starting from 0.

        Args:
            solution: Partition with possibly sparse group IDs

        Returns:
            Partition with groups numbered 0, 1, 2, ...
        """
        groups_used = sorted(set(solution))
        mapping = {old: new for new, old in enumerate(groups_used)}
        return [mapping[g] for g in solution]

    def repair_empty_groups(self, solution: list[int]) -> list[int]:
        """Remove empty groups by compacting.

        Args:
            solution: Partition

        Returns:
            Compacted partition
        """
        return self.compact_groups(solution)

    # =========================================================================
    # Universal methods (required by OptimizationContext)
    # =========================================================================

    def random_solution(self) -> list[int]:
        """Generate a random valid partition.

        Default: assigns each item to a random group.
        Override for domain-specific constraints.
        """
        return [random.randint(0, self.n_groups - 1) for _ in range(self.n_items)]

    def copy(self, solution: list[int]) -> list[int]:
        """Deep copy of partition."""
        return solution.copy()

    def valid(self, solution: list[int]) -> bool:
        """Check if solution is a valid partition.

        Default: checks length and group IDs within range.
        Override for domain-specific constraints.
        """
        if len(solution) != self.n_items:
            return False
        return all(0 <= g < self.n_groups for g in solution)

    # =========================================================================
    # Abstract methods (must be implemented by domain contexts)
    # =========================================================================

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'bin_packing', 'graph_partition')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of items to partition."""

    @abstractmethod
    def evaluate(self, solution: list[int]) -> float:
        """Evaluate partition cost (domain-specific)."""


__all__ = ["PartitionContext"]
