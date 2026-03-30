"""Base operators for binary optimization problems.

11 operators covering the standard role taxonomy:
- Construction (4): greedy, random, deterministic, hybrid
- Local Search (4): first improvement, best improvement, swap, multi-flip
- Perturbation (3): random, guided, adaptive

These operators work with any BinaryContext subclass (Knapsack, Set Cover, etc.)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.geakg.contexts.families.binary import BinaryContext


# =============================================================================
# CONSTRUCTION OPERATORS (4)
# =============================================================================


def const_greedy(solution: list[int] | None, ctx: "BinaryContext") -> list[int]:
    """Greedy construction by value/cost ratio.

    Adds items in decreasing order of their benefit, stopping when
    infeasible. Uses delta_flip to estimate item value.

    Args:
        solution: Ignored (starts from scratch)
        ctx: Binary context with evaluate/valid methods

    Returns:
        Constructed solution
    """
    n = ctx.dimension
    result = [0] * n

    # Calculate benefit of adding each item
    benefits: list[tuple[int, float]] = []
    for i in range(n):
        candidate = result.copy()
        candidate[i] = 1
        if ctx.valid(candidate):
            # Benefit = improvement in objective (negative delta = good)
            delta = ctx.evaluate(candidate) - ctx.evaluate(result)
            benefits.append((i, -delta))  # Higher benefit = better

    # Sort by benefit (descending)
    benefits.sort(key=lambda x: x[1], reverse=True)

    # Greedily add items
    for i, _ in benefits:
        candidate = result.copy()
        candidate[i] = 1
        if ctx.valid(candidate):
            result = candidate

    return result


def const_random(solution: list[int] | None, ctx: "BinaryContext") -> list[int]:
    """Random construction with validity check.

    Randomly selects items and keeps adding until infeasible.

    Args:
        solution: Ignored
        ctx: Binary context

    Returns:
        Random valid solution
    """
    n = ctx.dimension
    result = [0] * n
    indices = list(range(n))
    random.shuffle(indices)

    for i in indices:
        candidate = result.copy()
        candidate[i] = 1
        if ctx.valid(candidate):
            result = candidate

    return result


def const_deterministic(solution: list[int] | None, ctx: "BinaryContext") -> list[int]:
    """Deterministic construction by index order.

    Adds items in order 0, 1, 2, ... while maintaining feasibility.
    Provides reproducible baseline.

    Args:
        solution: Ignored
        ctx: Binary context

    Returns:
        Deterministically constructed solution
    """
    n = ctx.dimension
    result = [0] * n

    for i in range(n):
        candidate = result.copy()
        candidate[i] = 1
        if ctx.valid(candidate):
            result = candidate

    return result


def const_hybrid(solution: list[int] | None, ctx: "BinaryContext") -> list[int]:
    """Hybrid construction: greedy with randomization.

    Uses restricted candidate list (RCL) like GRASP.

    Args:
        solution: Ignored
        ctx: Binary context

    Returns:
        Solution with greedy+random construction
    """
    n = ctx.dimension
    result = [0] * n
    alpha = 0.3  # RCL threshold

    remaining = set(range(n))

    while remaining:
        # Calculate benefit for each remaining item
        benefits: list[tuple[int, float]] = []
        for i in remaining:
            candidate = result.copy()
            candidate[i] = 1
            if ctx.valid(candidate):
                delta = ctx.evaluate(candidate) - ctx.evaluate(result)
                benefits.append((i, -delta))

        if not benefits:
            break

        # Build RCL
        benefits.sort(key=lambda x: x[1], reverse=True)
        max_benefit = benefits[0][1]
        min_benefit = benefits[-1][1]
        threshold = max_benefit - alpha * (max_benefit - min_benefit)

        rcl = [i for i, b in benefits if b >= threshold]

        # Select randomly from RCL
        selected = random.choice(rcl)
        result[selected] = 1
        remaining.remove(selected)

    return result


# =============================================================================
# LOCAL SEARCH OPERATORS (4)
# =============================================================================


def ls_first_improvement(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """First improvement local search (single flip).

    Accepts the first improving flip found.

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Improved solution or original if no improvement
    """
    n = len(solution)
    current_cost = ctx.evaluate(solution)

    indices = list(range(n))
    random.shuffle(indices)

    for i in indices:
        candidate = ctx.flip(solution, i)
        if ctx.valid(candidate) and ctx.evaluate(candidate) < current_cost:
            return candidate

    return solution


def ls_best_improvement(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """Best improvement local search (single flip).

    Evaluates all flips and takes the best one.

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Best neighbor or original if no improvement
    """
    n = len(solution)
    best = solution
    best_cost = ctx.evaluate(solution)

    for i in range(n):
        candidate = ctx.flip(solution, i)
        if ctx.valid(candidate):
            cost = ctx.evaluate(candidate)
            if cost < best_cost:
                best = candidate
                best_cost = cost

    return best


def ls_swap(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """Swap-based local search.

    Tries swapping a 0 with a 1 (maintains cardinality).

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Improved solution or original
    """
    zeros = ctx.get_unselected_indices(solution)
    ones = ctx.get_selected_indices(solution)

    if not zeros or not ones:
        return solution

    current_cost = ctx.evaluate(solution)
    best = solution
    best_cost = current_cost

    # Sample to avoid O(n^2)
    sample_size = min(20, len(zeros), len(ones))
    zeros_sample = random.sample(zeros, sample_size) if len(zeros) > sample_size else zeros
    ones_sample = random.sample(ones, sample_size) if len(ones) > sample_size else ones

    for i in zeros_sample:
        for j in ones_sample:
            candidate = solution.copy()
            candidate[i] = 1
            candidate[j] = 0
            if ctx.valid(candidate):
                cost = ctx.evaluate(candidate)
                if cost < best_cost:
                    best = candidate
                    best_cost = cost

    return best


def ls_multi_flip(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """Multi-flip local search (2-flip neighborhood).

    Tries flipping pairs of bits for deeper search.

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Improved solution or original
    """
    n = len(solution)
    current_cost = ctx.evaluate(solution)
    best = solution
    best_cost = current_cost

    # Sample pairs to avoid O(n^2)
    indices = list(range(n))
    num_samples = min(100, n * (n - 1) // 2)

    for _ in range(num_samples):
        i, j = random.sample(indices, 2)
        candidate = ctx.flip_multiple(solution, [i, j])
        if ctx.valid(candidate):
            cost = ctx.evaluate(candidate)
            if cost < best_cost:
                best = candidate
                best_cost = cost

    return best


# =============================================================================
# PERTURBATION OPERATORS (3)
# =============================================================================


def pert_random(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """Random perturbation (random flips).

    Flips a random number of bits to escape local optimum.

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Perturbed solution
    """
    n = len(solution)
    num_flips = random.randint(1, max(1, n // 10))
    indices = random.sample(range(n), num_flips)

    result = ctx.flip_multiple(solution, indices)

    # Repair if invalid
    if not ctx.valid(result):
        result = ctx.repair_greedy(result)

    return result


def pert_guided(solution: list[int], ctx: "BinaryContext") -> list[int]:
    """Guided perturbation based on item costs.

    Flips bits that contribute most to cost (for selected items)
    or least (for unselected items).

    Args:
        solution: Current solution
        ctx: Binary context

    Returns:
        Perturbed solution
    """
    n = len(solution)
    result = solution.copy()

    # Find items with worst contribution
    ones = ctx.get_selected_indices(solution)
    zeros = ctx.get_unselected_indices(solution)

    # Calculate flip impact for each
    flip_impacts: list[tuple[int, float]] = []
    for i in range(n):
        candidate = ctx.flip(solution, i)
        if ctx.valid(candidate):
            delta = ctx.evaluate(candidate) - ctx.evaluate(solution)
            # For ones: positive delta is bad (removing hurts)
            # For zeros: negative delta is good (adding helps)
            impact = delta if solution[i] == 1 else -delta
            flip_impacts.append((i, impact))

    if not flip_impacts:
        return pert_random(solution, ctx)

    # Sort by impact (flip items where it helps or doesn't hurt much)
    flip_impacts.sort(key=lambda x: x[1])

    # Flip top few
    num_flips = random.randint(1, max(1, len(flip_impacts) // 5))
    for i, _ in flip_impacts[:num_flips]:
        result = ctx.flip(result, i)
        if not ctx.valid(result):
            result = ctx.repair_greedy(result)
            break

    return result


def pert_adaptive(
    solution: list[int],
    ctx: "BinaryContext",
    intensity: float = 0.1,
) -> list[int]:
    """Adaptive perturbation with configurable intensity.

    Intensity controls how much of the solution to perturb.

    Args:
        solution: Current solution
        ctx: Binary context
        intensity: Fraction of bits to flip (0.0 to 1.0)

    Returns:
        Perturbed solution
    """
    n = len(solution)
    num_flips = max(1, int(n * intensity))

    indices = random.sample(range(n), num_flips)
    result = ctx.flip_multiple(solution, indices)

    # Repair if invalid
    if not ctx.valid(result):
        result = ctx.repair_greedy(result)

    return result


# =============================================================================
# OPERATOR REGISTRY
# =============================================================================

BASE_OPERATORS_BINARY: dict[str, Callable] = {
    # Construction
    "const_greedy": const_greedy,
    "const_random": const_random,
    "const_deterministic": const_deterministic,
    "const_hybrid": const_hybrid,
    # Local Search
    "ls_first_improvement": ls_first_improvement,
    "ls_best_improvement": ls_best_improvement,
    "ls_swap": ls_swap,
    "ls_multi_flip": ls_multi_flip,
    # Perturbation
    "pert_random": pert_random,
    "pert_guided": pert_guided,
    "pert_adaptive": pert_adaptive,
}


__all__ = [
    "BASE_OPERATORS_BINARY",
    # Construction
    "const_greedy",
    "const_random",
    "const_deterministic",
    "const_hybrid",
    # Local Search
    "ls_first_improvement",
    "ls_best_improvement",
    "ls_swap",
    "ls_multi_flip",
    # Perturbation
    "pert_random",
    "pert_guided",
    "pert_adaptive",
]
