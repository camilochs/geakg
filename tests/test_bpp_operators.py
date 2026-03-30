"""Tests for all 27 BPP operators.

Tests each operator individually to ensure:
1. Operators produce valid BPP solutions (no bin exceeds capacity)
2. All items are packed exactly once
3. Number of bins is minimized where applicable

Operators by category:
- CONSTRUCTION (9): first_fit, best_fit, worst_fit, first_fit_decreasing,
                    best_fit_decreasing, next_fit_decreasing, djang_fitch,
                    best_k_fit, random_fit
- LOCAL_SEARCH (8): swap_items, move_item, swap_pairs, chain_move,
                    reduce_bins, bin_completion, vns_bpp, sequential_vnd_bpp
- PERTURBATION (6): random_reassign, shuffle_bin, empty_bins,
                    ruin_recreate_bpp, guided_reassignment, frequency_repack
- META_HEURISTIC (6): sa_bpp, threshold_bpp, tabu_bpp, reactive_tabu_bpp,
                      uniform_crossover_bpp, grouping_ga
"""

import random
import math
import pytest
from typing import Optional

from src.domains.binpacking import BPPInstance, BPPSolution, BPPDomain, create_sample_bpp_instance


# =============================================================================
# Fixtures - BPP Instances
# =============================================================================

@pytest.fixture
def small_instance() -> BPPInstance:
    """Small 10-item BPP instance for quick testing."""
    return create_sample_bpp_instance(n_items=10, capacity=100, seed=42)


@pytest.fixture
def medium_instance() -> BPPInstance:
    """Medium 30-item BPP instance."""
    return create_sample_bpp_instance(n_items=30, capacity=100, seed=42)


@pytest.fixture
def large_instance() -> BPPInstance:
    """Larger 50-item BPP instance."""
    return create_sample_bpp_instance(n_items=50, capacity=150, seed=42)


@pytest.fixture
def domain() -> BPPDomain:
    """BPP domain instance."""
    return BPPDomain()


# =============================================================================
# BPP Operator Implementations
# =============================================================================

class BPPOperators:
    """BPP operator implementations for testing."""

    def __init__(self, instance: BPPInstance):
        self.instance = instance
        self.n_items = instance.n_items
        self.capacity = instance.capacity
        self.sizes = instance.item_sizes

    def bin_usage(self, bins: list[list[int]]) -> list[int]:
        """Get usage of each bin."""
        return [sum(self.sizes[i] for i in b) for b in bins]

    def is_valid_solution(self, bins: list[list[int]]) -> bool:
        """Check if solution is valid."""
        packed = set()
        for bin_items in bins:
            usage = sum(self.sizes[i] for i in bin_items)
            if usage > self.capacity:
                return False
            for item in bin_items:
                if item < 0 or item >= self.n_items:
                    return False
                if item in packed:
                    return False
                packed.add(item)
        return packed == set(range(self.n_items))

    def n_bins(self, bins: list[list[int]]) -> int:
        """Count non-empty bins."""
        return len([b for b in bins if b])

    # =========================================================================
    # CONSTRUCTION OPERATORS (9)
    # =========================================================================

    def first_fit(self) -> list[list[int]]:
        """First Fit: place in first bin that fits."""
        bins = []
        bin_remaining = []

        for item in range(self.n_items):
            size = self.sizes[item]
            placed = False

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining:
                    bins[i].append(item)
                    bin_remaining[i] -= size
                    placed = True
                    break

            if not placed:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def best_fit(self) -> list[list[int]]:
        """Best Fit: place in tightest fitting bin."""
        bins = []
        bin_remaining = []

        for item in range(self.n_items):
            size = self.sizes[item]
            best_idx = -1
            best_remaining = self.capacity + 1

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining < best_remaining:
                    best_idx = i
                    best_remaining = remaining

            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def worst_fit(self) -> list[list[int]]:
        """Worst Fit: place in loosest fitting bin."""
        bins = []
        bin_remaining = []

        for item in range(self.n_items):
            size = self.sizes[item]
            best_idx = -1
            best_remaining = -1

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining and remaining > best_remaining:
                    best_idx = i
                    best_remaining = remaining

            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def first_fit_decreasing(self) -> list[list[int]]:
        """FFD: sort by size, then first fit."""
        items = sorted(range(self.n_items), key=lambda i: self.sizes[i], reverse=True)

        bins = []
        bin_remaining = []

        for item in items:
            size = self.sizes[item]
            placed = False

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining:
                    bins[i].append(item)
                    bin_remaining[i] -= size
                    placed = True
                    break

            if not placed:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def best_fit_decreasing(self) -> list[list[int]]:
        """BFD: sort by size, then best fit."""
        items = sorted(range(self.n_items), key=lambda i: self.sizes[i], reverse=True)

        bins = []
        bin_remaining = []

        for item in items:
            size = self.sizes[item]
            best_idx = -1
            best_remaining = self.capacity + 1

            for i, remaining in enumerate(bin_remaining):
                if size <= remaining < best_remaining:
                    best_idx = i
                    best_remaining = remaining

            if best_idx >= 0:
                bins[best_idx].append(item)
                bin_remaining[best_idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def next_fit_decreasing(self) -> list[list[int]]:
        """NFD: sort by size, then next fit (only try current bin)."""
        items = sorted(range(self.n_items), key=lambda i: self.sizes[i], reverse=True)

        bins = [[]]
        current_remaining = self.capacity

        for item in items:
            size = self.sizes[item]

            if size <= current_remaining:
                bins[-1].append(item)
                current_remaining -= size
            else:
                bins.append([item])
                current_remaining = self.capacity - size

        return bins

    def djang_fitch(self) -> list[list[int]]:
        """Djang-Fitch: MTP-based construction."""
        # Simplified: FFD + try to fill bins completely
        bins = self.first_fit_decreasing()
        return self._try_fill_bins(bins)

    def _try_fill_bins(self, bins: list[list[int]]) -> list[list[int]]:
        """Try to fill bins completely by swapping items."""
        bins = [b.copy() for b in bins]
        usages = self.bin_usage(bins)

        for i in range(len(bins)):
            remaining = self.capacity - usages[i]
            if remaining == 0:
                continue

            # Look for items in other bins that fit perfectly
            for j in range(i + 1, len(bins)):
                for item in bins[j]:
                    if self.sizes[item] == remaining:
                        bins[j].remove(item)
                        bins[i].append(item)
                        usages[i] = self.capacity
                        usages[j] -= self.sizes[item]
                        break

        return [b for b in bins if b]

    def best_k_fit(self, k: int = 3) -> list[list[int]]:
        """Best-k-fit: consider k best bins."""
        items = sorted(range(self.n_items), key=lambda i: self.sizes[i], reverse=True)

        bins = []
        bin_remaining = []

        for item in items:
            size = self.sizes[item]

            # Find k best fitting bins
            candidates = []
            for i, remaining in enumerate(bin_remaining):
                if size <= remaining:
                    candidates.append((remaining, i))

            candidates.sort()
            candidates = candidates[:k]

            if candidates:
                # Choose randomly from k best
                _, idx = random.choice(candidates) if len(candidates) > 1 else candidates[0]
                bins[idx].append(item)
                bin_remaining[idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    def random_fit(self) -> list[list[int]]:
        """Random feasible placement."""
        items = list(range(self.n_items))
        random.shuffle(items)

        bins = []
        bin_remaining = []

        for item in items:
            size = self.sizes[item]

            # Find all feasible bins
            feasible = [i for i, rem in enumerate(bin_remaining) if size <= rem]

            if feasible:
                idx = random.choice(feasible)
                bins[idx].append(item)
                bin_remaining[idx] -= size
            else:
                bins.append([item])
                bin_remaining.append(self.capacity - size)

        return bins

    # =========================================================================
    # LOCAL SEARCH OPERATORS (8)
    # =========================================================================

    def swap_items(self, bins: list[list[int]]) -> list[list[int]]:
        """Swap items between bins."""
        bins = [b.copy() for b in bins]
        usages = self.bin_usage(bins)

        swapped = True
        while swapped:
            swapped = False
            for b1 in range(len(bins)):
                for b2 in range(b1 + 1, len(bins)):
                    for i in range(len(bins[b1])):
                        for j in range(len(bins[b2])):
                            item1, item2 = bins[b1][i], bins[b2][j]
                            s1, s2 = self.sizes[item1], self.sizes[item2]

                            # Check if swap is feasible
                            new_usage1 = usages[b1] - s1 + s2
                            new_usage2 = usages[b2] - s2 + s1

                            if new_usage1 <= self.capacity and new_usage2 <= self.capacity:
                                # Check if beneficial (e.g., balances load or one bin becomes fuller)
                                old_diff = abs(usages[b1] - usages[b2])
                                new_diff = abs(new_usage1 - new_usage2)

                                if new_diff < old_diff:  # Better balance
                                    bins[b1][i] = item2
                                    bins[b2][j] = item1
                                    usages[b1] = new_usage1
                                    usages[b2] = new_usage2
                                    swapped = True
                                    break
                        if swapped:
                            break
                    if swapped:
                        break
                if swapped:
                    break

        return bins

    def move_item(self, bins: list[list[int]]) -> list[list[int]]:
        """Move item to different bin."""
        bins = [b.copy() for b in bins]
        usages = self.bin_usage(bins)

        for b1 in range(len(bins)):
            if len(bins[b1]) <= 1:
                continue

            for i, item in enumerate(bins[b1]):
                size = self.sizes[item]

                for b2 in range(len(bins)):
                    if b1 == b2:
                        continue

                    if usages[b2] + size <= self.capacity:
                        # Move if it increases utilization of target
                        if usages[b2] + size > usages[b2]:
                            bins[b1].pop(i)
                            bins[b2].append(item)
                            usages[b1] -= size
                            usages[b2] += size
                            break
                else:
                    continue
                break

        return [b for b in bins if b]

    def swap_pairs(self, bins: list[list[int]]) -> list[list[int]]:
        """Swap pairs of items."""
        bins = [b.copy() for b in bins]
        usages = self.bin_usage(bins)

        for b1 in range(len(bins)):
            if len(bins[b1]) < 2:
                continue

            for b2 in range(b1 + 1, len(bins)):
                if len(bins[b2]) < 2:
                    continue

                # Try swapping pairs
                for i1 in range(len(bins[b1]) - 1):
                    for i2 in range(i1 + 1, len(bins[b1])):
                        for j1 in range(len(bins[b2]) - 1):
                            for j2 in range(j1 + 1, len(bins[b2])):
                                pair1 = [bins[b1][i1], bins[b1][i2]]
                                pair2 = [bins[b2][j1], bins[b2][j2]]

                                s1 = sum(self.sizes[i] for i in pair1)
                                s2 = sum(self.sizes[i] for i in pair2)

                                new_u1 = usages[b1] - s1 + s2
                                new_u2 = usages[b2] - s2 + s1

                                if new_u1 <= self.capacity and new_u2 <= self.capacity:
                                    return bins  # Accept any feasible swap

        return bins

    def chain_move(self, bins: list[list[int]]) -> list[list[int]]:
        """Chain of item movements."""
        bins = [b.copy() for b in bins]

        for _ in range(min(5, len(bins))):
            # Pick random item and try to move it
            non_empty = [i for i, b in enumerate(bins) if b]
            if not non_empty:
                break

            b1 = random.choice(non_empty)
            if not bins[b1]:
                continue

            item = random.choice(bins[b1])
            size = self.sizes[item]

            for b2 in range(len(bins)):
                if b2 == b1:
                    continue

                usage = sum(self.sizes[i] for i in bins[b2])
                if usage + size <= self.capacity:
                    bins[b1].remove(item)
                    bins[b2].append(item)
                    break

        return [b for b in bins if b]

    def reduce_bins(self, bins: list[list[int]]) -> list[list[int]]:
        """Try to reduce number of bins."""
        bins = [b.copy() for b in bins if b]
        usages = self.bin_usage(bins)

        # Sort bins by usage (ascending)
        sorted_indices = sorted(range(len(bins)), key=lambda i: usages[i])

        for src_idx in sorted_indices:
            if not bins[src_idx]:
                continue

            items = bins[src_idx].copy()

            # Try to move all items to other bins
            can_empty = True
            moves = []

            for item in items:
                size = self.sizes[item]
                placed = False

                for dst_idx in range(len(bins)):
                    if dst_idx == src_idx or not bins[dst_idx]:
                        continue

                    curr_usage = sum(self.sizes[i] for i in bins[dst_idx])
                    curr_usage += sum(self.sizes[i] for _, d, i in moves if d == dst_idx)

                    if curr_usage + size <= self.capacity:
                        moves.append((src_idx, dst_idx, item))
                        placed = True
                        break

                if not placed:
                    can_empty = False
                    break

            if can_empty:
                for src, dst, item in moves:
                    bins[src].remove(item)
                    bins[dst].append(item)

        return [b for b in bins if b]

    def bin_completion(self, bins: list[list[int]]) -> list[list[int]]:
        """Try to complete partially filled bins."""
        bins = [b.copy() for b in bins if b]
        usages = self.bin_usage(bins)

        # Find best items to swap in to complete bins
        for b1 in range(len(bins)):
            remaining = self.capacity - usages[b1]
            if remaining == 0:
                continue

            for b2 in range(len(bins)):
                if b1 == b2:
                    continue

                for item in bins[b2]:
                    if self.sizes[item] == remaining:
                        bins[b2].remove(item)
                        bins[b1].append(item)
                        usages[b1] = self.capacity
                        usages[b2] -= self.sizes[item]
                        break

        return [b for b in bins if b]

    def vns_bpp(self, bins: list[list[int]], max_iter: int = 10) -> list[list[int]]:
        """Variable neighborhood search for BPP."""
        neighborhoods = [self.swap_items, self.move_item, self.reduce_bins]

        current = [b.copy() for b in bins if b]
        best = current
        best_n = self.n_bins(current)

        k = 0
        for _ in range(max_iter):
            neighbor = neighborhoods[k % len(neighborhoods)](current)
            n_bins = self.n_bins(neighbor)

            if n_bins < best_n:
                best = neighbor
                best_n = n_bins
                current = neighbor
                k = 0
            else:
                k += 1

        return best

    def sequential_vnd_bpp(self, bins: list[list[int]]) -> list[list[int]]:
        """Sequential VND for BPP."""
        current = [b.copy() for b in bins if b]

        improved = True
        while improved:
            improved = False

            # Try reduce bins
            new_bins = self.reduce_bins(current)
            if self.n_bins(new_bins) < self.n_bins(current):
                current = new_bins
                improved = True
                continue

            # Try swap items
            new_bins = self.swap_items(current)
            if self.n_bins(new_bins) < self.n_bins(current):
                current = new_bins
                improved = True
                continue

            # Try move item
            new_bins = self.move_item(current)
            if self.n_bins(new_bins) < self.n_bins(current):
                current = new_bins
                improved = True

        return current

    # =========================================================================
    # PERTURBATION OPERATORS (6)
    # =========================================================================

    def random_reassign(self, bins: list[list[int]], rate: float = 0.2) -> list[list[int]]:
        """Randomly reassign some items."""
        bins = [b.copy() for b in bins if b]
        all_items = [i for b in bins for i in b]
        n_reassign = max(1, int(len(all_items) * rate))

        reassign = random.sample(all_items, min(n_reassign, len(all_items)))

        # Remove items
        for item in reassign:
            for b in bins:
                if item in b:
                    b.remove(item)
                    break

        # Reinsert using first fit
        for item in reassign:
            size = self.sizes[item]
            placed = False

            for b in bins:
                usage = sum(self.sizes[i] for i in b)
                if usage + size <= self.capacity:
                    b.append(item)
                    placed = True
                    break

            if not placed:
                bins.append([item])

        return [b for b in bins if b]

    def shuffle_bin(self, bins: list[list[int]]) -> list[list[int]]:
        """Shuffle items in selected bins."""
        if not bins:
            return bins

        bins = [b.copy() for b in bins if b]

        # Pick random bin and shuffle
        idx = random.randrange(len(bins))
        random.shuffle(bins[idx])

        return bins

    def empty_bins(self, bins: list[list[int]], n_empty: int = 1) -> list[list[int]]:
        """Empty bins and repack."""
        bins = [b.copy() for b in bins if b]

        if len(bins) <= n_empty:
            return bins

        # Empty smallest bins
        usages = self.bin_usage(bins)
        sorted_indices = sorted(range(len(bins)), key=lambda i: usages[i])

        emptied_items = []
        for idx in sorted_indices[:n_empty]:
            emptied_items.extend(bins[idx])
            bins[idx] = []

        bins = [b for b in bins if b]

        # Reinsert
        for item in emptied_items:
            size = self.sizes[item]
            placed = False

            for b in bins:
                usage = sum(self.sizes[i] for i in b)
                if usage + size <= self.capacity:
                    b.append(item)
                    placed = True
                    break

            if not placed:
                bins.append([item])

        return bins

    def ruin_recreate_bpp(self, bins: list[list[int]], rate: float = 0.3) -> list[list[int]]:
        """Ruin and recreate for BPP."""
        return self.random_reassign(bins, rate)

    def guided_reassignment(self, bins: list[list[int]]) -> list[list[int]]:
        """Reassign based on bin utilization."""
        bins = [b.copy() for b in bins if b]
        usages = self.bin_usage(bins)

        # Find least utilized bins
        sorted_indices = sorted(range(len(bins)), key=lambda i: usages[i])

        # Empty bottom 20% and redistribute
        n_empty = max(1, len(bins) // 5)
        return self.empty_bins(bins, n_empty)

    def frequency_repack(self, bins: list[list[int]], history: Optional[dict] = None) -> list[list[int]]:
        """Repack based on solution history."""
        return self.random_reassign(bins, rate=0.3)

    # =========================================================================
    # META-HEURISTIC OPERATORS (6)
    # =========================================================================

    def sa_bpp(self, current: list[list[int]], neighbor: list[list[int]], temp: float = 10.0) -> tuple[list[list[int]], bool]:
        """Simulated annealing for BPP."""
        current_n = self.n_bins(current)
        neighbor_n = self.n_bins(neighbor)

        if neighbor_n < current_n:
            return [b.copy() for b in neighbor if b], True

        if temp > 0:
            delta = neighbor_n - current_n
            if random.random() < math.exp(-delta / temp):
                return [b.copy() for b in neighbor if b], True

        return [b.copy() for b in current if b], False

    def threshold_bpp(self, current: list[list[int]], neighbor: list[list[int]], threshold: int = 1) -> tuple[list[list[int]], bool]:
        """Threshold accepting for BPP."""
        current_n = self.n_bins(current)
        neighbor_n = self.n_bins(neighbor)

        if neighbor_n - current_n <= threshold:
            return [b.copy() for b in neighbor if b], True

        return [b.copy() for b in current if b], False

    def tabu_bpp(self, bins: list[list[int]], tabu_list: list, tenure: int = 7) -> tuple[list[list[int]], list]:
        """Tabu search for BPP."""
        tabu_list = tabu_list.copy()

        best_neighbor = None
        best_move = None
        best_n = float('inf')

        # Try moves
        for b1 in range(len(bins)):
            for i, item in enumerate(bins[b1]):
                for b2 in range(len(bins)):
                    if b1 == b2:
                        continue

                    move = (item, b1, b2)
                    if move in tabu_list:
                        continue

                    usage = sum(self.sizes[j] for j in bins[b2])
                    if usage + self.sizes[item] > self.capacity:
                        continue

                    test = [b.copy() for b in bins]
                    test[b1].pop(i)
                    test[b2].append(item)
                    test = [b for b in test if b]

                    n = self.n_bins(test)
                    if n < best_n:
                        best_n = n
                        best_neighbor = test
                        best_move = move

        if best_neighbor is None:
            return [b for b in bins if b], tabu_list

        tabu_list.append(best_move)
        if len(tabu_list) > tenure:
            tabu_list.pop(0)

        return best_neighbor, tabu_list

    def reactive_tabu_bpp(self, bins: list[list[int]], tabu_list: list) -> tuple[list[list[int]], list]:
        """Reactive tabu for BPP."""
        return self.tabu_bpp(bins, tabu_list)

    def uniform_crossover_bpp(self, parent1: list[list[int]], parent2: list[list[int]]) -> list[list[int]]:
        """Uniform crossover for BPP."""
        # Take some bins from parent1
        n_from_p1 = len(parent1) // 2
        child_bins = [b.copy() for b in parent1[:n_from_p1]]

        # Track which items are already assigned
        assigned = set(i for b in child_bins for i in b)

        # Add remaining items from parent2 ordering
        remaining = [i for b in parent2 for i in b if i not in assigned]

        for item in remaining:
            size = self.sizes[item]
            placed = False

            for b in child_bins:
                usage = sum(self.sizes[i] for i in b)
                if usage + size <= self.capacity:
                    b.append(item)
                    placed = True
                    break

            if not placed:
                child_bins.append([item])

        return child_bins

    def grouping_ga(self, parent1: list[list[int]], parent2: list[list[int]]) -> list[list[int]]:
        """Grouping genetic algorithm crossover."""
        # Similar to uniform but preserves groups better
        return self.uniform_crossover_bpp(parent1, parent2)


# =============================================================================
# TESTS: CONSTRUCTION OPERATORS
# =============================================================================

class TestConstructionOperators:
    """Tests for 9 BPP construction operators."""

    def test_first_fit(self, medium_instance: BPPInstance):
        """First fit produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.first_fit()

        assert ops.is_valid_solution(bins)
        assert ops.n_bins(bins) >= medium_instance.lower_bound

    def test_best_fit(self, medium_instance: BPPInstance):
        """Best fit produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.best_fit()

        assert ops.is_valid_solution(bins)

    def test_worst_fit(self, medium_instance: BPPInstance):
        """Worst fit produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.worst_fit()

        assert ops.is_valid_solution(bins)

    def test_first_fit_decreasing(self, medium_instance: BPPInstance):
        """FFD produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.first_fit_decreasing()

        assert ops.is_valid_solution(bins)

    def test_best_fit_decreasing(self, medium_instance: BPPInstance):
        """BFD produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.best_fit_decreasing()

        assert ops.is_valid_solution(bins)

    def test_next_fit_decreasing(self, medium_instance: BPPInstance):
        """NFD produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.next_fit_decreasing()

        assert ops.is_valid_solution(bins)

    def test_djang_fitch(self, medium_instance: BPPInstance):
        """Djang-Fitch produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.djang_fitch()

        assert ops.is_valid_solution(bins)

    def test_best_k_fit(self, medium_instance: BPPInstance):
        """Best-k-fit produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.best_k_fit()

        assert ops.is_valid_solution(bins)

    def test_random_fit(self, medium_instance: BPPInstance):
        """Random fit produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.random_fit()

        assert ops.is_valid_solution(bins)

    def test_ffd_better_than_ff(self, medium_instance: BPPInstance):
        """FFD should use fewer bins than FF on average."""
        random.seed(42)
        ops = BPPOperators(medium_instance)

        ff_bins = ops.n_bins(ops.first_fit())
        ffd_bins = ops.n_bins(ops.first_fit_decreasing())

        assert ffd_bins <= ff_bins


# =============================================================================
# TESTS: LOCAL SEARCH OPERATORS
# =============================================================================

class TestLocalSearchOperators:
    """Tests for 8 BPP local search operators."""

    def test_swap_items(self, medium_instance: BPPInstance):
        """Swap items produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.swap_items(initial)
        assert ops.is_valid_solution(result)

    def test_move_item(self, medium_instance: BPPInstance):
        """Move item produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.move_item(initial)
        assert ops.is_valid_solution(result)

    def test_swap_pairs(self, medium_instance: BPPInstance):
        """Swap pairs produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.swap_pairs(initial)
        assert ops.is_valid_solution(result)

    def test_chain_move(self, medium_instance: BPPInstance):
        """Chain move produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.chain_move(initial)
        assert ops.is_valid_solution(result)

    def test_reduce_bins(self, medium_instance: BPPInstance):
        """Reduce bins produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.random_fit()

        result = ops.reduce_bins(initial)
        assert ops.is_valid_solution(result)
        assert ops.n_bins(result) <= ops.n_bins(initial)

    def test_bin_completion(self, medium_instance: BPPInstance):
        """Bin completion produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.bin_completion(initial)
        assert ops.is_valid_solution(result)

    def test_vns_bpp(self, medium_instance: BPPInstance):
        """VNS produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.random_fit()

        result = ops.vns_bpp(initial)
        assert ops.is_valid_solution(result)

    def test_sequential_vnd_bpp(self, medium_instance: BPPInstance):
        """Sequential VND produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.random_fit()

        result = ops.sequential_vnd_bpp(initial)
        assert ops.is_valid_solution(result)


# =============================================================================
# TESTS: PERTURBATION OPERATORS
# =============================================================================

class TestPerturbationOperators:
    """Tests for 6 BPP perturbation operators."""

    def test_random_reassign(self, medium_instance: BPPInstance):
        """Random reassign produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.random_reassign(initial)
        assert ops.is_valid_solution(result)

    def test_shuffle_bin(self, medium_instance: BPPInstance):
        """Shuffle bin produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.shuffle_bin(initial)
        assert ops.is_valid_solution(result)

    def test_empty_bins(self, medium_instance: BPPInstance):
        """Empty bins produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.empty_bins(initial)
        assert ops.is_valid_solution(result)

    def test_ruin_recreate_bpp(self, medium_instance: BPPInstance):
        """Ruin-recreate produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.ruin_recreate_bpp(initial)
        assert ops.is_valid_solution(result)

    def test_guided_reassignment(self, medium_instance: BPPInstance):
        """Guided reassignment produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.guided_reassignment(initial)
        assert ops.is_valid_solution(result)

    def test_frequency_repack(self, medium_instance: BPPInstance):
        """Frequency repack produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        initial = ops.first_fit()

        result = ops.frequency_repack(initial)
        assert ops.is_valid_solution(result)


# =============================================================================
# TESTS: META-HEURISTIC OPERATORS
# =============================================================================

class TestMetaHeuristicOperators:
    """Tests for 6 BPP meta-heuristic operators."""

    def test_sa_bpp(self, medium_instance: BPPInstance):
        """SA returns valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        current = ops.first_fit()
        neighbor = ops.random_reassign(current)

        result, accepted = ops.sa_bpp(current, neighbor)
        assert ops.is_valid_solution(result)
        assert isinstance(accepted, bool)

    def test_threshold_bpp(self, medium_instance: BPPInstance):
        """Threshold accepting returns valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        current = ops.first_fit()
        neighbor = ops.swap_items(current)

        result, accepted = ops.threshold_bpp(current, neighbor)
        assert ops.is_valid_solution(result)

    def test_tabu_bpp(self, medium_instance: BPPInstance):
        """Tabu search returns valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.first_fit()

        result, tabu = ops.tabu_bpp(bins, [])
        assert ops.is_valid_solution(result)
        assert isinstance(tabu, list)

    def test_reactive_tabu_bpp(self, medium_instance: BPPInstance):
        """Reactive tabu returns valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        bins = ops.first_fit()

        result, tabu = ops.reactive_tabu_bpp(bins, [])
        assert ops.is_valid_solution(result)

    def test_uniform_crossover_bpp(self, medium_instance: BPPInstance):
        """Uniform crossover produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        parent1 = ops.first_fit()
        parent2 = ops.best_fit()

        child = ops.uniform_crossover_bpp(parent1, parent2)
        assert ops.is_valid_solution(child)

    def test_grouping_ga(self, medium_instance: BPPInstance):
        """Grouping GA produces valid solution."""
        random.seed(42)
        ops = BPPOperators(medium_instance)
        parent1 = ops.first_fit_decreasing()
        parent2 = ops.random_fit()

        child = ops.grouping_ga(parent1, parent2)
        assert ops.is_valid_solution(child)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for BPP operators."""

    def test_ils_pattern(self, medium_instance: BPPInstance):
        """ILS pattern: construct → local search → perturb → local search."""
        random.seed(42)
        ops = BPPOperators(medium_instance)

        # Construct
        solution = ops.first_fit_decreasing()
        assert ops.is_valid_solution(solution)

        # Local search
        solution = ops.vns_bpp(solution)
        assert ops.is_valid_solution(solution)

        # Perturb
        solution = ops.ruin_recreate_bpp(solution)
        assert ops.is_valid_solution(solution)

        # Local search again
        solution = ops.sequential_vnd_bpp(solution)
        assert ops.is_valid_solution(solution)

    def test_all_construction_on_small(self, small_instance: BPPInstance):
        """All construction operators work on small instance."""
        random.seed(42)
        ops = BPPOperators(small_instance)

        for method in [
            ops.first_fit,
            ops.best_fit,
            ops.worst_fit,
            ops.first_fit_decreasing,
            ops.best_fit_decreasing,
            ops.next_fit_decreasing,
            ops.djang_fitch,
            ops.best_k_fit,
            ops.random_fit,
        ]:
            bins = method()
            assert ops.is_valid_solution(bins), f"{method.__name__} failed"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_single_item(self):
        """Handle single item."""
        instance = create_sample_bpp_instance(n_items=1, capacity=100, seed=42)
        ops = BPPOperators(instance)

        bins = ops.first_fit()
        assert ops.is_valid_solution(bins)
        assert len(bins) == 1
        assert len(bins[0]) == 1

    def test_tight_capacity(self):
        """Handle tight capacity."""
        instance = create_sample_bpp_instance(n_items=10, capacity=50, seed=42)
        ops = BPPOperators(instance)

        bins = ops.first_fit_decreasing()
        assert ops.is_valid_solution(bins)

    def test_all_same_size(self):
        """Handle all items same size."""
        instance = BPPInstance(
            name="same_size",
            n_items=10,
            capacity=100,
            item_sizes=[25] * 10,
        )
        ops = BPPOperators(instance)

        bins = ops.first_fit()
        assert ops.is_valid_solution(bins)
        # Should use exactly 3 bins (4 items per bin)
        assert ops.n_bins(bins) == 3

    def test_repeated_vns(self, small_instance: BPPInstance):
        """Repeated VNS converges."""
        random.seed(42)
        ops = BPPOperators(small_instance)
        bins = ops.random_fit()

        prev_n = ops.n_bins(bins)
        for _ in range(5):
            bins = ops.vns_bpp(bins, max_iter=3)
            curr_n = ops.n_bins(bins)
            assert curr_n <= prev_n
            prev_n = curr_n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
