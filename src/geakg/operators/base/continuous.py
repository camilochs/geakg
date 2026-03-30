"""Base operators for continuous optimization problems.

11 operators covering the standard role taxonomy:
- Construction (4): random, latin hypercube, quasi-random, centroid
- Local Search (4): gradient descent, pattern search, Nelder-Mead step, coordinate descent
- Perturbation (3): Gaussian, uniform, adaptive

These operators work with any ContinuousContext subclass.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.geakg.contexts.families.continuous import ContinuousContext


# =============================================================================
# CONSTRUCTION OPERATORS (4)
# =============================================================================


def const_random(solution: list[float] | None, ctx: "ContinuousContext") -> list[float]:
    """Random construction within bounds.

    Generates a uniformly random point in the search space.

    Args:
        solution: Ignored
        ctx: Continuous context with bounds

    Returns:
        Random solution
    """
    return ctx.random_solution()


def const_latin_hypercube(
    solution: list[float] | None, ctx: "ContinuousContext"
) -> list[float]:
    """Latin Hypercube-inspired single sample.

    Divides each dimension into intervals and samples from random intervals.

    Args:
        solution: Ignored
        ctx: Continuous context

    Returns:
        LHS-style solution
    """
    n = ctx.dimension
    result = []
    n_intervals = 10  # Number of intervals per dimension

    for i in range(n):
        lower, upper = ctx.bounds[i]
        interval_size = (upper - lower) / n_intervals
        interval = random.randint(0, n_intervals - 1)
        value = lower + interval * interval_size + random.random() * interval_size
        result.append(value)

    return result


def const_quasi_random(
    solution: list[float] | None, ctx: "ContinuousContext", seed: int = 0
) -> list[float]:
    """Quasi-random construction using Halton-like sequence.

    Uses low-discrepancy sequence for better space coverage.

    Args:
        solution: Ignored
        ctx: Continuous context
        seed: Seed for sequence position

    Returns:
        Quasi-random solution
    """
    def halton(index: int, base: int) -> float:
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = ctx.dimension

    result = []
    for i in range(n):
        base = primes[i % len(primes)]
        h = halton(seed + random.randint(1, 1000), base)
        lower, upper = ctx.bounds[i]
        result.append(lower + h * (upper - lower))

    return result


def const_centroid(
    solution: list[float] | None, ctx: "ContinuousContext"
) -> list[float]:
    """Centroid construction (middle of bounds).

    Places solution at the center of the search space.
    Good starting point for unimodal functions.

    Args:
        solution: Ignored
        ctx: Continuous context

    Returns:
        Centroid solution
    """
    return [(lower + upper) / 2 for lower, upper in ctx.bounds]


# =============================================================================
# LOCAL SEARCH OPERATORS (4)
# =============================================================================


def ls_gradient_descent(
    solution: list[float], ctx: "ContinuousContext", step_size: float = 0.01
) -> list[float]:
    """Gradient descent step.

    Takes one step in the negative gradient direction.

    Args:
        solution: Current solution
        ctx: Continuous context
        step_size: Learning rate

    Returns:
        Improved solution
    """
    gradient = ctx.gradient(solution)
    return ctx.gradient_step(solution, gradient, step_size)


def ls_pattern_search(
    solution: list[float], ctx: "ContinuousContext", delta: float = 0.1
) -> list[float]:
    """Pattern search (coordinate-wise probing).

    Probes along each coordinate axis and accepts improvements.

    Args:
        solution: Current solution
        ctx: Continuous context
        delta: Step size (relative to range)

    Returns:
        Improved solution
    """
    result = solution.copy()
    current_cost = ctx.evaluate(result)

    for i in range(ctx.dimension):
        lower, upper = ctx.bounds[i]
        step = delta * (upper - lower)

        # Try positive direction
        candidate = result.copy()
        candidate[i] = min(upper, result[i] + step)
        cost = ctx.evaluate(candidate)
        if cost < current_cost:
            result = candidate
            current_cost = cost
            continue

        # Try negative direction
        candidate = result.copy()
        candidate[i] = max(lower, result[i] - step)
        cost = ctx.evaluate(candidate)
        if cost < current_cost:
            result = candidate
            current_cost = cost

    return result


def ls_nelder_mead_step(
    solution: list[float], ctx: "ContinuousContext"
) -> list[float]:
    """Single Nelder-Mead-inspired step.

    Creates a small simplex around the point and moves toward
    the best vertex direction.

    Args:
        solution: Current solution
        ctx: Continuous context

    Returns:
        Improved solution
    """
    n = ctx.dimension
    delta = 0.05  # Simplex size

    # Create simplex vertices
    vertices = [solution.copy()]
    for i in range(n):
        v = solution.copy()
        lower, upper = ctx.bounds[i]
        v[i] += delta * (upper - lower)
        v = ctx.clip(v)
        vertices.append(v)

    # Evaluate all vertices
    costs = [(v, ctx.evaluate(v)) for v in vertices]
    costs.sort(key=lambda x: x[1])

    best = costs[0][0]
    worst = costs[-1][0]

    # Reflect worst through centroid
    centroid = [
        sum(v[i] for v, _ in costs[:-1]) / n
        for i in range(n)
    ]

    reflected = [2 * centroid[i] - worst[i] for i in range(n)]
    reflected = ctx.clip(reflected)

    if ctx.evaluate(reflected) < costs[0][1]:
        return reflected
    else:
        return best


def ls_coordinate_descent(
    solution: list[float], ctx: "ContinuousContext", n_iter: int = 5
) -> list[float]:
    """Coordinate descent with line search.

    Optimizes one coordinate at a time using golden section search.

    Args:
        solution: Current solution
        ctx: Continuous context
        n_iter: Iterations per coordinate

    Returns:
        Improved solution
    """
    result = solution.copy()
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    for i in range(ctx.dimension):
        lower, upper = ctx.bounds[i]

        # Golden section search on coordinate i
        a, b = lower, upper
        c = b - (b - a) / phi
        d = a + (b - a) / phi

        for _ in range(n_iter):
            candidate_c = result.copy()
            candidate_c[i] = c
            candidate_d = result.copy()
            candidate_d[i] = d

            if ctx.evaluate(candidate_c) < ctx.evaluate(candidate_d):
                b = d
            else:
                a = c

            c = b - (b - a) / phi
            d = a + (b - a) / phi

        result[i] = (a + b) / 2

    return result


# =============================================================================
# PERTURBATION OPERATORS (3)
# =============================================================================


def pert_gaussian(
    solution: list[float], ctx: "ContinuousContext", sigma: float = 0.1
) -> list[float]:
    """Gaussian perturbation.

    Adds Gaussian noise to all dimensions.

    Args:
        solution: Current solution
        ctx: Continuous context
        sigma: Standard deviation (relative to range)

    Returns:
        Perturbed solution
    """
    return ctx.perturb(solution, sigma)


def pert_uniform(
    solution: list[float], ctx: "ContinuousContext", delta: float = 0.2
) -> list[float]:
    """Uniform perturbation.

    Adds uniform noise to all dimensions.

    Args:
        solution: Current solution
        ctx: Continuous context
        delta: Maximum perturbation (relative to range)

    Returns:
        Perturbed solution
    """
    return ctx.perturb_uniform(solution, delta)


def pert_adaptive(
    solution: list[float],
    ctx: "ContinuousContext",
    intensity: float = 0.1,
) -> list[float]:
    """Adaptive perturbation.

    Perturbs a subset of dimensions with varying intensity.

    Args:
        solution: Current solution
        ctx: Continuous context
        intensity: Controls perturbation strength

    Returns:
        Perturbed solution
    """
    n = ctx.dimension

    # Perturb a random subset of dimensions
    num_dims = max(1, int(n * intensity))
    indices = random.sample(range(n), num_dims)

    # Use varying sigma for diversity
    sigma = random.uniform(0.05, 0.2)

    return ctx.perturb(solution, sigma, indices)


# =============================================================================
# OPERATOR REGISTRY
# =============================================================================

BASE_OPERATORS_CONTINUOUS: dict[str, Callable] = {
    # Construction
    "const_random": const_random,
    "const_latin_hypercube": const_latin_hypercube,
    "const_quasi_random": const_quasi_random,
    "const_centroid": const_centroid,
    # Local Search
    "ls_gradient_descent": ls_gradient_descent,
    "ls_pattern_search": ls_pattern_search,
    "ls_nelder_mead_step": ls_nelder_mead_step,
    "ls_coordinate_descent": ls_coordinate_descent,
    # Perturbation
    "pert_gaussian": pert_gaussian,
    "pert_uniform": pert_uniform,
    "pert_adaptive": pert_adaptive,
}


__all__ = [
    "BASE_OPERATORS_CONTINUOUS",
    # Construction
    "const_random",
    "const_latin_hypercube",
    "const_quasi_random",
    "const_centroid",
    # Local Search
    "ls_gradient_descent",
    "ls_pattern_search",
    "ls_nelder_mead_step",
    "ls_coordinate_descent",
    # Perturbation
    "pert_gaussian",
    "pert_uniform",
    "pert_adaptive",
]
