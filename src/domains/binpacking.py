"""Bin Packing Problem (BPP) domain implementation.

Supports:
- 1D Bin Packing Problem
- Minimize number of bins used
"""

import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from src.domains.base import OptimizationDomain, ProblemFeatures


class BPPInstance(BaseModel):
    """Bin Packing Problem instance.

    A BPP instance consists of:
    - A set of items with sizes
    - Bin capacity
    - Goal: pack all items in minimum number of bins
    """

    name: str
    n_items: int = Field(gt=0)
    capacity: int = Field(gt=0)
    # item_sizes[i] = size of item i
    item_sizes: list[int]
    optimal_bins: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def dimension(self) -> int:
        """Number of items."""
        return self.n_items

    @computed_field
    @property
    def total_size(self) -> int:
        """Total size of all items."""
        return sum(self.item_sizes)

    @computed_field
    @property
    def lower_bound(self) -> int:
        """Lower bound on number of bins (continuous relaxation)."""
        import math
        return math.ceil(self.total_size / self.capacity)


class BPPSolution(BaseModel):
    """BPP solution (bin assignments).

    Each bin is a list of item indices.
    """

    bins: list[list[int]]  # bins[i] = list of items in bin i
    cost: float = Field(ge=0, default=0.0)  # Number of bins used
    is_valid: bool = True

    @property
    def n_bins(self) -> int:
        """Number of bins used."""
        return len([b for b in self.bins if b])

    @property
    def all_items(self) -> set[int]:
        """Set of all packed items."""
        return {item for bin in self.bins for item in bin}


class BPPFeatures(ProblemFeatures):
    """Features extracted from BPP instance."""

    dimension: int
    n_items: int
    capacity: int
    total_size: int
    avg_item_size: float
    std_item_size: float
    max_item_size: int
    min_item_size: int
    fill_ratio: float  # total_size / (lower_bound * capacity)
    large_item_ratio: float  # fraction of items > capacity/2

    @classmethod
    def from_instance(cls, instance: BPPInstance) -> "BPPFeatures":
        """Extract features from BPP instance."""
        sizes = instance.item_sizes
        n = len(sizes)

        avg_size = sum(sizes) / n if n > 0 else 0
        variance = sum((s - avg_size) ** 2 for s in sizes) / n if n > 0 else 0
        std_size = variance ** 0.5

        large_items = sum(1 for s in sizes if s > instance.capacity / 2)
        large_ratio = large_items / n if n > 0 else 0

        lb = instance.lower_bound
        fill_ratio = instance.total_size / (lb * instance.capacity) if lb > 0 else 0

        return cls(
            dimension=n,
            n_items=n,
            capacity=instance.capacity,
            total_size=instance.total_size,
            avg_item_size=avg_size,
            std_item_size=std_size,
            max_item_size=max(sizes) if sizes else 0,
            min_item_size=min(sizes) if sizes else 0,
            fill_ratio=fill_ratio,
            large_item_ratio=large_ratio,
        )


class BPPDomain(OptimizationDomain[BPPInstance, BPPSolution]):
    """BPP domain implementation."""

    @property
    def name(self) -> str:
        return "bpp"

    def load_instance(self, path: Path) -> BPPInstance:
        """Load BPP instance from standard format.

        Supports OR-Library and BPPLib formats.

        Args:
            path: Path to instance file

        Returns:
            Loaded BPP instance
        """
        with open(path) as f:
            content = f.read()

        return self._parse_bpplib(content, path.stem)

    def _parse_bpplib(self, content: str, name: str) -> BPPInstance:
        """Parse BPPLib format content.

        Format:
        Line 1: n_items capacity
        Lines 2+: one item size per line
        """
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]

        # First line: n_items capacity (or just n_items in some formats)
        first_parts = lines[0].split()
        n_items = int(first_parts[0])
        capacity = int(first_parts[1]) if len(first_parts) > 1 else 100

        # Read item sizes
        item_sizes = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                item_sizes.append(int(parts[0]))
            if len(item_sizes) >= n_items:
                break

        return BPPInstance(
            name=name,
            n_items=n_items,
            capacity=capacity,
            item_sizes=item_sizes,
        )

    def evaluate_solution(self, solution: BPPSolution, instance: BPPInstance) -> float:
        """Compute number of bins used.

        Args:
            solution: BPP solution
            instance: BPP instance

        Returns:
            Number of bins (lower is better)
        """
        return float(solution.n_bins)

    def validate_solution(self, solution: BPPSolution, instance: BPPInstance) -> bool:
        """Check if solution is valid.

        Validates:
        - All items are packed exactly once
        - No bin exceeds capacity

        Args:
            solution: BPP solution
            instance: BPP instance

        Returns:
            True if valid
        """
        packed = set()

        for bin_items in solution.bins:
            bin_size = 0
            for item in bin_items:
                # Check valid item index
                if item < 0 or item >= instance.n_items:
                    return False
                # Check not packed before
                if item in packed:
                    return False
                packed.add(item)
                bin_size += instance.item_sizes[item]

            # Check capacity constraint
            if bin_size > instance.capacity:
                return False

        # Check all items packed
        if packed != set(range(instance.n_items)):
            return False

        return True

    def get_features(self, instance: BPPInstance) -> BPPFeatures:
        """Extract features from BPP instance.

        Args:
            instance: BPP instance

        Returns:
            Extracted features
        """
        return BPPFeatures.from_instance(instance)

    def random_solution(self, instance: BPPInstance) -> BPPSolution:
        """Generate a random valid solution.

        Uses random assignment respecting capacity.

        Args:
            instance: BPP instance

        Returns:
            Random valid solution
        """
        items = list(range(instance.n_items))
        random.shuffle(items)

        bins = []
        current_bin = []
        current_size = 0

        for item in items:
            size = instance.item_sizes[item]
            if current_size + size <= instance.capacity:
                current_bin.append(item)
                current_size += size
            else:
                if current_bin:
                    bins.append(current_bin)
                current_bin = [item]
                current_size = size

        if current_bin:
            bins.append(current_bin)

        solution = BPPSolution(bins=bins)
        solution.cost = float(len(bins))
        return solution

    def first_fit(self, instance: BPPInstance) -> BPPSolution:
        """Generate solution using First Fit algorithm.

        Args:
            instance: BPP instance

        Returns:
            First Fit solution
        """
        bins = []
        bin_remaining = []

        for item in range(instance.n_items):
            size = instance.item_sizes[item]

            # Find first bin that fits
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if size <= remaining:
                    bins[i].append(item)
                    bin_remaining[i] -= size
                    placed = True
                    break

            if not placed:
                bins.append([item])
                bin_remaining.append(instance.capacity - size)

        solution = BPPSolution(bins=bins)
        solution.cost = float(len(bins))
        return solution

    def first_fit_decreasing(self, instance: BPPInstance) -> BPPSolution:
        """Generate solution using First Fit Decreasing algorithm.

        Args:
            instance: BPP instance

        Returns:
            FFD solution
        """
        # Sort items by size (decreasing)
        items = sorted(range(instance.n_items),
                       key=lambda i: instance.item_sizes[i],
                       reverse=True)

        bins = []
        bin_remaining = []

        for item in items:
            size = instance.item_sizes[item]

            # Find first bin that fits
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if size <= remaining:
                    bins[i].append(item)
                    bin_remaining[i] -= size
                    placed = True
                    break

            if not placed:
                bins.append([item])
                bin_remaining.append(instance.capacity - size)

        solution = BPPSolution(bins=bins)
        solution.cost = float(len(bins))
        return solution

    def best_fit(self, instance: BPPInstance) -> BPPSolution:
        """Generate solution using Best Fit algorithm.

        Args:
            instance: BPP instance

        Returns:
            Best Fit solution
        """
        bins = []
        bin_remaining = []

        for item in range(instance.n_items):
            size = instance.item_sizes[item]

            # Find bin with minimum remaining space that fits
            best_idx = -1
            best_remaining = instance.capacity + 1

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining < best_remaining:
                    best_idx = i
                    best_remaining = remaining

            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(instance.capacity - size)

        solution = BPPSolution(bins=bins)
        solution.cost = float(len(bins))
        return solution

    def best_fit_decreasing(self, instance: BPPInstance) -> BPPSolution:
        """Generate solution using Best Fit Decreasing algorithm.

        Args:
            instance: BPP instance

        Returns:
            BFD solution
        """
        # Sort items by size (decreasing)
        items = sorted(range(instance.n_items),
                       key=lambda i: instance.item_sizes[i],
                       reverse=True)

        bins = []
        bin_remaining = []

        for item in items:
            size = instance.item_sizes[item]

            # Find bin with minimum remaining space that fits
            best_idx = -1
            best_remaining = instance.capacity + 1

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining < best_remaining:
                    best_idx = i
                    best_remaining = remaining

            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(instance.capacity - size)

        solution = BPPSolution(bins=bins)
        solution.cost = float(len(bins))
        return solution


def create_sample_bpp_instance(
    n_items: int = 20,
    capacity: int = 100,
    seed: int = 42,
) -> BPPInstance:
    """Create a sample BPP instance for testing.

    Args:
        n_items: Number of items
        capacity: Bin capacity
        seed: Random seed

    Returns:
        Sample BPP instance
    """
    random.seed(seed)

    # Generate item sizes (between 10% and 50% of capacity)
    min_size = max(1, capacity // 10)
    max_size = capacity // 2
    item_sizes = [random.randint(min_size, max_size) for _ in range(n_items)]

    return BPPInstance(
        name=f"sample_bpp_{n_items}",
        n_items=n_items,
        capacity=capacity,
        item_sizes=item_sizes,
    )
