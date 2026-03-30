"""Generic operators for permutation-based problems.

These operators work on any problem where the solution is a permutation
of integers [0, 1, ..., n-1]. Examples: TSP, JSSP, QAP, PFSP.

Total: 11 operators covering 11 abstract roles (3 categories).
- Construction: 4 operators
- Local Search: 4 operators
- Perturbation: 3 operators
"""

import random
from typing import Any, Callable

from ..representations import (
    GenericOperator,
    RepresentationOperators,
    RepresentationType,
)


# =============================================================================
# CONSTRUCTION OPERATORS (4)
# =============================================================================


def greedy_by_fitness(
    n: int,
    partial_fitness_fn: Callable[[list[int], int], float],
) -> list[int]:
    """Construct permutation by greedily adding best element.

    At each step, adds the element that minimizes the partial fitness.
    This is a generic greedy construction that works for any permutation problem.

    Args:
        n: Size of the permutation.
        partial_fitness_fn: Function(current_perm, candidate) -> fitness delta.

    Returns:
        Complete permutation.
    """
    remaining = set(range(n))
    perm: list[int] = []

    while remaining:
        best_elem = min(remaining, key=lambda x: partial_fitness_fn(perm, x))
        perm.append(best_elem)
        remaining.remove(best_elem)

    return perm


def random_insertion_construct(n: int) -> list[int]:
    """Construct permutation by inserting elements at random positions.

    Args:
        n: Size of the permutation.

    Returns:
        Random permutation built incrementally.
    """
    perm: list[int] = []
    elements = list(range(n))
    random.shuffle(elements)

    for elem in elements:
        if not perm:
            perm.append(elem)
        else:
            pos = random.randint(0, len(perm))
            perm.insert(pos, elem)

    return perm


def pairwise_merge_construct(
    n: int,
    merge_gain_fn: Callable[[list[int], list[int]], float] | None = None,
) -> list[int]:
    """Construct permutation by merging pairs with highest gain.

    Starts with n singleton lists, iteratively merges best pairs.

    Args:
        n: Size of the permutation.
        merge_gain_fn: Optional function to compute merge gain.

    Returns:
        Complete permutation.
    """
    # Start with singletons
    fragments: list[list[int]] = [[i] for i in range(n)]

    while len(fragments) > 1:
        # Find best pair to merge (random if no gain function)
        if merge_gain_fn is None:
            i, j = random.sample(range(len(fragments)), 2)
        else:
            best_gain = float("-inf")
            best_i, best_j = 0, 1
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    gain = merge_gain_fn(fragments[i], fragments[j])
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
            i, j = best_i, best_j

        # Merge fragments[i] and fragments[j]
        merged = fragments[i] + fragments[j]
        fragments = [f for k, f in enumerate(fragments) if k not in (i, j)]
        fragments.append(merged)

    return fragments[0]


def random_permutation_construct(n: int) -> list[int]:
    """Construct a completely random permutation.

    Args:
        n: Size of the permutation.

    Returns:
        Random permutation.
    """
    perm = list(range(n))
    random.shuffle(perm)
    return perm


# =============================================================================
# LOCAL SEARCH OPERATORS (8)
# =============================================================================


def swap(perm: list[int], i: int | None = None, j: int | None = None) -> list[int]:
    """Swap elements at positions i and j.

    Args:
        perm: Current permutation.
        i: First position (random if None).
        j: Second position (random if None).

    Returns:
        New permutation with swapped elements.
    """
    n = len(perm)
    if n < 2:
        return perm.copy()

    if i is None:
        i = random.randint(0, n - 1)
    if j is None:
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)

    result = perm.copy()
    result[i], result[j] = result[j], result[i]
    return result


def _insert(perm: list[int], i: int | None = None, j: int | None = None) -> list[int]:
    """Remove element at position i and insert at position j (internal helper)."""
    n = len(perm)
    if n < 2:
        return perm.copy()

    if i is None:
        i = random.randint(0, n - 1)
    if j is None:
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)

    result = perm.copy()
    elem = result.pop(i)
    result.insert(j, elem)
    return result


def _invert(perm: list[int], i: int | None = None, j: int | None = None) -> list[int]:
    """Invert (reverse) segment from position i to j (internal helper).

    Args:
        perm: Current permutation.
        i: Start of segment (random if None).
        j: End of segment (random if None).

    Returns:
        New permutation with segment reversed.
    """
    n = len(perm)
    if n < 2:
        return perm.copy()

    if i is None or j is None:
        i, j = sorted(random.sample(range(n), 2))

    if i > j:
        i, j = j, i

    result = perm.copy()
    result[i : j + 1] = reversed(result[i : j + 1])
    return result


def segment_reverse(
    perm: list[int], i: int | None = None, length: int | None = None
) -> list[int]:
    """Reverse a segment of specified length starting at position i.

    Args:
        perm: Current permutation.
        i: Start position (random if None).
        length: Segment length (random 3-n/3 if None).

    Returns:
        New permutation with segment reversed.
    """
    n = len(perm)
    if n < 3:
        return perm.copy()

    if length is None:
        max_len = max(3, n // 3)
        length = random.randint(3, max_len)

    if i is None:
        i = random.randint(0, n - length)

    j = min(i + length - 1, n - 1)
    return _invert(perm, i, j)


def variable_depth_search(
    perm: list[int],
    fitness_fn: Callable[[list[int]], float],
    max_depth: int = 5,
) -> list[int]:
    """Basic first-improvement local search.

    Tries random swaps until no improvement found.
    Intentionally simple - synthesis should synthesize better operators.

    Args:
        perm: Current permutation.
        fitness_fn: Function to evaluate permutation fitness.
        max_depth: Maximum iterations without improvement.

    Returns:
        Improved permutation.
    """
    current = perm.copy()
    current_fitness = fitness_fn(current)
    n = len(current)

    if n < 2:
        return current

    no_improve = 0
    while no_improve < max_depth:
        # Try random swap
        i, j = random.sample(range(n), 2)
        candidate = swap(current, i, j)
        if fitness_fn(candidate) < current_fitness:
            current = candidate
            current_fitness = fitness_fn(current)
            no_improve = 0
        else:
            no_improve += 1

    return current


def vnd_generic(
    perm: list[int],
    fitness_fn: Callable[[list[int]], float],
    max_iterations: int = 10,
) -> list[int]:
    """Basic Variable Neighborhood Descent.

    Cycles through swap and invert neighborhoods.
    Intentionally simple - synthesis should synthesize better VND operators.

    Args:
        perm: Current permutation.
        fitness_fn: Function to evaluate permutation fitness.
        max_iterations: Maximum total iterations.

    Returns:
        Improved permutation.
    """
    current = perm.copy()
    current_fitness = fitness_fn(current)
    n = len(current)

    if n < 2:
        return current

    for _ in range(max_iterations):
        i, j = random.sample(range(n), 2)

        # Try swap
        candidate = swap(current, i, j)
        if fitness_fn(candidate) < current_fitness:
            current = candidate
            current_fitness = fitness_fn(current)
            continue

        # Try invert if swap didn't improve
        if n >= 4:
            i, j = min(i, j), max(i, j)
            candidate = _invert(current, i, j)
            if fitness_fn(candidate) < current_fitness:
                current = candidate
                current_fitness = fitness_fn(current)

    return current


# =============================================================================
# PERTURBATION OPERATORS (3)
# =============================================================================


def segment_shuffle(perm: list[int], k: int | None = None) -> list[int]:
    """Shuffle a random segment of size k.

    Args:
        perm: Current permutation.
        k: Segment size (random 3-n/4 if None).

    Returns:
        Perturbed permutation.
    """
    n = len(perm)
    if n < 3:
        return perm.copy()

    if k is None:
        k = random.randint(3, max(3, n // 4))

    k = min(k, n)
    start = random.randint(0, n - k)

    result = perm.copy()
    segment = result[start : start + k]
    random.shuffle(segment)
    result[start : start + k] = segment
    return result


def partial_restart(perm: list[int], ratio: float = 0.3) -> list[int]:
    """Randomly reinitialize a portion of the permutation.

    Args:
        perm: Current permutation.
        ratio: Fraction of elements to reinitialize.

    Returns:
        Perturbed permutation.
    """
    n = len(perm)
    k = max(2, int(n * ratio))

    # Select positions to restart
    positions = random.sample(range(n), k)
    values = [perm[p] for p in positions]
    random.shuffle(values)

    result = perm.copy()
    for pos, val in zip(positions, values):
        result[pos] = val
    return result


def history_guided_perturb(
    perm: list[int],
    history: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Perturb based on move history (avoid recent moves).

    Args:
        perm: Current permutation.
        history: List of recent (i, j) swap positions to avoid.

    Returns:
        Perturbed permutation avoiding recent moves.
    """
    n = len(perm)
    if n < 2:
        return perm.copy()

    # Avoid positions used in recent history
    avoid_positions: set[int] = set()
    if history:
        for i, j in history[-10:]:  # Last 10 moves
            avoid_positions.add(i)
            avoid_positions.add(j)

    # Find positions not in history
    available = [p for p in range(n) if p not in avoid_positions]
    if len(available) < 2:
        available = list(range(n))

    # Apply multiple random swaps on non-history positions
    result = perm.copy()
    num_swaps = random.randint(2, max(2, n // 10))
    for _ in range(num_swaps):
        if len(available) >= 2:
            i, j = random.sample(available, 2)
            result[i], result[j] = result[j], result[i]

    return result


# =============================================================================
# REGISTER ALL OPERATORS
# =============================================================================


def _create_permutation_operators() -> RepresentationOperators:
    """Create the collection of all permutation operators."""
    ops = RepresentationOperators(representation=RepresentationType.PERMUTATION)

    # Construction (4)
    ops.add_operator(
        GenericOperator(
            operator_id="greedy_by_fitness",
            function=greedy_by_fitness,
            role="const_greedy",
            weight=1.0,
            description="Greedy construction minimizing partial fitness",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="random_insertion",
            function=random_insertion_construct,
            role="const_insertion",
            weight=1.0,
            description="Insert elements at random positions",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="pairwise_merge",
            function=pairwise_merge_construct,
            role="const_savings",
            weight=1.0,
            description="Merge pairs by gain (savings-style)",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="random_permutation",
            function=random_permutation_construct,
            role="const_random",
            weight=1.0,
            description="Completely random permutation",
        )
    )

    # Local Search (4)
    ops.add_operator(
        GenericOperator(
            operator_id="swap",
            function=swap,
            role="ls_intensify_small",
            weight=1.0,
            description="Swap two elements",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="segment_reverse",
            function=segment_reverse,
            role="ls_intensify_medium",
            weight=1.0,
            description="Reverse longer segment",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="variable_depth_search",
            function=variable_depth_search,
            role="ls_intensify_large",
            weight=1.0,
            description="Variable depth swap-based search",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="vnd_generic",
            function=vnd_generic,
            role="ls_chain",
            weight=1.0,
            description="Variable Neighborhood Descent",
        )
    )

    # Perturbation (3)
    ops.add_operator(
        GenericOperator(
            operator_id="segment_shuffle",
            function=segment_shuffle,
            role="pert_escape_small",
            weight=1.0,
            description="Shuffle contiguous segment",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="partial_restart",
            function=partial_restart,
            role="pert_escape_large",
            weight=1.0,
            description="Reinitialize portion of solution",
        )
    )
    ops.add_operator(
        GenericOperator(
            operator_id="history_guided_perturb",
            function=history_guided_perturb,
            role="pert_adaptive",
            weight=1.0,
            description="Perturb avoiding recent moves",
        )
    )

    return ops


# Singleton instance
PERMUTATION_OPERATORS = _create_permutation_operators()
