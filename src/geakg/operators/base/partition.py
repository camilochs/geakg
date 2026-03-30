"""Base operators for partition optimization problems.

11 operators covering the standard role taxonomy:
- Construction (4): first fit, best fit, random, hybrid
- Local Search (4): single move, swap, chain, multi-move
- Perturbation (3): random, guided, adaptive

These operators work with any PartitionContext subclass (Bin Packing, Clustering, etc.)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.geakg.contexts.families.partition import PartitionContext


# =============================================================================
# CONSTRUCTION OPERATORS (4)
# =============================================================================


def const_first_fit(
    solution: list[int] | None, ctx: "PartitionContext"
) -> list[int]:
    """First Fit construction.

    Assigns each item to the first group that can accept it.
    Groups are created as needed.

    Args:
        solution: Ignored
        ctx: Partition context

    Returns:
        First-fit solution
    """
    n = ctx.n_items
    result = [-1] * n  # -1 means unassigned
    next_group = 0

    for item in range(n):
        # Try existing groups
        assigned = False
        for group in range(next_group):
            candidate = result.copy()
            candidate[item] = group
            if ctx.valid(candidate):
                result[item] = group
                assigned = True
                break

        # Create new group if needed
        if not assigned:
            result[item] = next_group
            next_group += 1

    return result


def const_best_fit(
    solution: list[int] | None, ctx: "PartitionContext"
) -> list[int]:
    """Best Fit construction.

    Assigns each item to the group where it fits best
    (e.g., minimum remaining capacity for bin packing).

    Args:
        solution: Ignored
        ctx: Partition context

    Returns:
        Best-fit solution
    """
    n = ctx.n_items
    result = [-1] * n
    next_group = 0

    for item in range(n):
        best_group = -1
        best_load = float("inf")

        # Find best existing group
        for group in range(next_group):
            candidate = result.copy()
            candidate[item] = group
            if ctx.valid(candidate):
                load = ctx.group_load(candidate, group)
                if load < best_load:
                    best_load = load
                    best_group = group

        if best_group >= 0:
            result[item] = best_group
        else:
            result[item] = next_group
            next_group += 1

    return result


def const_random(
    solution: list[int] | None, ctx: "PartitionContext"
) -> list[int]:
    """Random construction.

    Assigns items to random groups, creating new groups as needed.

    Args:
        solution: Ignored
        ctx: Partition context

    Returns:
        Random valid solution
    """
    n = ctx.n_items
    result = [-1] * n
    next_group = 0

    items = list(range(n))
    random.shuffle(items)

    for item in items:
        # Try random existing group
        if next_group > 0:
            groups = list(range(next_group))
            random.shuffle(groups)
            for group in groups:
                candidate = result.copy()
                candidate[item] = group
                if ctx.valid(candidate):
                    result[item] = group
                    break

        # Create new group if still unassigned
        if result[item] == -1:
            result[item] = next_group
            next_group += 1

    return result


def const_hybrid(
    solution: list[int] | None, ctx: "PartitionContext"
) -> list[int]:
    """Hybrid construction (GRASP-style).

    Uses restricted candidate list to balance greedy and random choices.

    Args:
        solution: Ignored
        ctx: Partition context

    Returns:
        Hybrid solution
    """
    n = ctx.n_items
    result = [-1] * n
    next_group = 0
    alpha = 0.3  # RCL threshold

    items = list(range(n))
    random.shuffle(items)

    for item in items:
        candidates: list[tuple[int, float]] = []

        # Evaluate all valid groups
        for group in range(next_group):
            candidate = result.copy()
            candidate[item] = group
            if ctx.valid(candidate):
                load = ctx.group_load(candidate, group)
                candidates.append((group, load))

        # Also consider new group
        candidates.append((next_group, 0.0))

        if candidates:
            # Build RCL
            loads = [load for _, load in candidates]
            min_load = min(loads)
            max_load = max(loads)
            threshold = min_load + alpha * (max_load - min_load)

            rcl = [(g, l) for g, l in candidates if l <= threshold]
            selected, _ = random.choice(rcl)

            result[item] = selected
            if selected == next_group:
                next_group += 1

    return result


# =============================================================================
# LOCAL SEARCH OPERATORS (4)
# =============================================================================


def ls_single_move(
    solution: list[int], ctx: "PartitionContext"
) -> list[int]:
    """Single move local search.

    Tries moving each item to a different group and accepts improvements.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Improved solution
    """
    result = solution.copy()
    current_cost = ctx.evaluate(result)
    improved = True

    while improved:
        improved = False
        items = list(range(ctx.n_items))
        random.shuffle(items)

        for item in items:
            current_group = result[item]
            groups = list(set(result) | {ctx.num_active_groups(result)})

            for group in groups:
                if group == current_group:
                    continue

                candidate = result.copy()
                candidate[item] = group
                if ctx.valid(candidate):
                    cost = ctx.evaluate(candidate)
                    if cost < current_cost:
                        result = candidate
                        current_cost = cost
                        improved = True
                        break

            if improved:
                break

    return result


def ls_swap(solution: list[int], ctx: "PartitionContext") -> list[int]:
    """Swap-based local search.

    Tries swapping group assignments between pairs of items.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Improved solution
    """
    result = solution.copy()
    current_cost = ctx.evaluate(result)
    n = ctx.n_items

    # Sample pairs to avoid O(n^2)
    num_samples = min(100, n * (n - 1) // 2)

    for _ in range(num_samples):
        i, j = random.sample(range(n), 2)
        if result[i] == result[j]:
            continue

        candidate = ctx.swap_items(result, i, j)
        if ctx.valid(candidate):
            cost = ctx.evaluate(candidate)
            if cost < current_cost:
                result = candidate
                current_cost = cost

    return result


def ls_chain(solution: list[int], ctx: "PartitionContext") -> list[int]:
    """Chain move local search.

    Performs a sequence of moves: item1 to group A, item2 to group B, etc.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Improved solution
    """
    result = solution.copy()
    current_cost = ctx.evaluate(result)

    max_chain_length = 3
    n = ctx.n_items

    for _ in range(n // 2):
        # Build a random chain
        chain_items = random.sample(range(n), min(max_chain_length, n))
        chain_groups = [result[item] for item in chain_items]

        # Rotate groups
        rotated_groups = chain_groups[1:] + [chain_groups[0]]

        candidate = result.copy()
        for item, group in zip(chain_items, rotated_groups):
            candidate[item] = group

        if ctx.valid(candidate):
            cost = ctx.evaluate(candidate)
            if cost < current_cost:
                result = candidate
                current_cost = cost

    return result


def ls_multi_move(
    solution: list[int], ctx: "PartitionContext"
) -> list[int]:
    """Multi-move local search.

    Tries moving multiple items from one group to another.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Improved solution
    """
    result = solution.copy()
    current_cost = ctx.evaluate(result)

    groups = ctx.get_groups(result)
    group_ids = list(groups.keys())

    if len(group_ids) < 2:
        return result

    # Try moving subsets between groups
    for _ in range(20):
        from_group = random.choice(group_ids)
        to_group = random.choice([g for g in group_ids if g != from_group])

        items_in_from = groups.get(from_group, [])
        if not items_in_from:
            continue

        # Move 1-3 items
        num_to_move = min(len(items_in_from), random.randint(1, 3))
        items_to_move = random.sample(items_in_from, num_to_move)

        candidate = result.copy()
        for item in items_to_move:
            candidate[item] = to_group

        if ctx.valid(candidate):
            cost = ctx.evaluate(candidate)
            if cost < current_cost:
                result = candidate
                current_cost = cost
                # Update groups for next iteration
                groups = ctx.get_groups(result)

    return result


# =============================================================================
# PERTURBATION OPERATORS (3)
# =============================================================================


def pert_random(solution: list[int], ctx: "PartitionContext") -> list[int]:
    """Random perturbation.

    Randomly moves some items to different groups.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Perturbed solution
    """
    result = solution.copy()
    n = ctx.n_items

    num_moves = random.randint(1, max(1, n // 5))
    items = random.sample(range(n), num_moves)

    active_groups = list(set(result))
    max_new_group = max(active_groups) + 1

    for item in items:
        # Move to random group (existing or new)
        new_group = random.randint(0, min(max_new_group, ctx.n_groups - 1))
        candidate = result.copy()
        candidate[item] = new_group
        if ctx.valid(candidate):
            result = candidate

    return ctx.compact_groups(result)


def pert_guided(solution: list[int], ctx: "PartitionContext") -> list[int]:
    """Guided perturbation.

    Moves items from overloaded groups to underloaded ones.

    Args:
        solution: Current solution
        ctx: Partition context

    Returns:
        Perturbed solution
    """
    result = solution.copy()
    groups = ctx.get_groups(result)

    if len(groups) < 2:
        return pert_random(result, ctx)

    # Find group loads
    loads = {g: ctx.group_load(result, g) for g in groups}
    avg_load = sum(loads.values()) / len(loads)

    # Identify overloaded and underloaded groups
    overloaded = [g for g, l in loads.items() if l > avg_load]
    underloaded = [g for g, l in loads.items() if l < avg_load]

    if not overloaded or not underloaded:
        return pert_random(result, ctx)

    # Move items from overloaded to underloaded
    num_moves = random.randint(1, 3)
    for _ in range(num_moves):
        from_group = random.choice(overloaded)
        to_group = random.choice(underloaded)

        items_in_from = groups.get(from_group, [])
        if items_in_from:
            item = random.choice(items_in_from)
            candidate = result.copy()
            candidate[item] = to_group
            if ctx.valid(candidate):
                result = candidate
                groups = ctx.get_groups(result)

    return result


def pert_adaptive(
    solution: list[int],
    ctx: "PartitionContext",
    intensity: float = 0.1,
) -> list[int]:
    """Adaptive perturbation.

    Perturbation strength scales with intensity parameter.

    Args:
        solution: Current solution
        ctx: Partition context
        intensity: Fraction of items to move

    Returns:
        Perturbed solution
    """
    result = solution.copy()
    n = ctx.n_items

    num_moves = max(1, int(n * intensity))
    items = random.sample(range(n), num_moves)

    active_groups = list(set(result))
    max_group = max(active_groups) if active_groups else 0

    for item in items:
        # Choose new group (favor existing groups)
        if random.random() < 0.8 and active_groups:
            new_group = random.choice(active_groups)
        else:
            new_group = max_group + 1
            if new_group >= ctx.n_groups:
                new_group = random.randint(0, ctx.n_groups - 1)

        candidate = result.copy()
        candidate[item] = new_group
        if ctx.valid(candidate):
            result = candidate
            active_groups = list(set(result))
            max_group = max(active_groups) if active_groups else 0

    return ctx.compact_groups(result)


# =============================================================================
# OPERATOR REGISTRY
# =============================================================================

BASE_OPERATORS_PARTITION: dict[str, Callable] = {
    # Construction
    "const_first_fit": const_first_fit,
    "const_best_fit": const_best_fit,
    "const_random": const_random,
    "const_hybrid": const_hybrid,
    # Local Search
    "ls_single_move": ls_single_move,
    "ls_swap": ls_swap,
    "ls_chain": ls_chain,
    "ls_multi_move": ls_multi_move,
    # Perturbation
    "pert_random": pert_random,
    "pert_guided": pert_guided,
    "pert_adaptive": pert_adaptive,
}


__all__ = [
    "BASE_OPERATORS_PARTITION",
    # Construction
    "const_first_fit",
    "const_best_fit",
    "const_random",
    "const_hybrid",
    # Local Search
    "ls_single_move",
    "ls_swap",
    "ls_chain",
    "ls_multi_move",
    # Perturbation
    "pert_random",
    "pert_guided",
    "pert_adaptive",
]
