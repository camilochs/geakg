"""Continuous Family Context.

Base class for all continuous optimization problems where
solutions are real-valued vectors [x1, x2, ..., xn].

Examples: Function Optimization, Neural Architecture Search, Hyperparameter Tuning
"""

from __future__ import annotations

from abc import abstractmethod
import random
import math
from typing import Optional

from src.geakg.contexts.base import FamilyContext, OptimizationFamily


class ContinuousContext(FamilyContext[list[float]]):
    """Context for continuous optimization.

    Solutions are represented as lists of floating-point numbers,
    typically bounded within specified ranges.

    Subclasses implement domain-specific evaluation and bounds.
    """

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.CONTINUOUS

    # =========================================================================
    # Bounds handling (critical for continuous optimization)
    # =========================================================================

    @property
    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """Get bounds for each dimension.

        Returns:
            List of (lower, upper) tuples, one per dimension
        """

    def clip(self, solution: list[float]) -> list[float]:
        """Clip solution to bounds.

        Args:
            solution: Potentially out-of-bounds solution

        Returns:
            Solution clipped to valid bounds
        """
        result = solution.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            result[i] = max(lower, min(upper, result[i]))
        return result

    def wrap(self, solution: list[float]) -> list[float]:
        """Wrap solution around bounds (toroidal).

        Args:
            solution: Potentially out-of-bounds solution

        Returns:
            Solution wrapped to valid bounds
        """
        result = solution.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            range_size = upper - lower
            if range_size <= 0:
                continue
            while result[i] < lower:
                result[i] += range_size
            while result[i] > upper:
                result[i] -= range_size
        return result

    def in_bounds(self, solution: list[float]) -> bool:
        """Check if solution is within bounds.

        Args:
            solution: Solution to check

        Returns:
            True if all values within bounds
        """
        for x, (lower, upper) in zip(solution, self.bounds):
            if x < lower or x > upper:
                return False
        return True

    # =========================================================================
    # Continuous-specific operations
    # =========================================================================

    def perturb(
        self, solution: list[float], sigma: float, indices: list[int] | None = None
    ) -> list[float]:
        """Perturb solution with Gaussian noise.

        Args:
            solution: Current solution
            sigma: Standard deviation of noise (relative to range)
            indices: Specific indices to perturb (None = all)

        Returns:
            Perturbed solution (clipped to bounds)
        """
        result = solution.copy()
        indices = indices if indices is not None else range(self.dimension)

        for i in indices:
            lower, upper = self.bounds[i]
            range_size = upper - lower
            result[i] += random.gauss(0, sigma * range_size)

        return self.clip(result)

    def perturb_uniform(
        self, solution: list[float], delta: float, indices: list[int] | None = None
    ) -> list[float]:
        """Perturb solution with uniform noise.

        Args:
            solution: Current solution
            delta: Maximum perturbation (relative to range)
            indices: Specific indices to perturb (None = all)

        Returns:
            Perturbed solution (clipped to bounds)
        """
        result = solution.copy()
        indices = indices if indices is not None else range(self.dimension)

        for i in indices:
            lower, upper = self.bounds[i]
            range_size = upper - lower
            result[i] += random.uniform(-delta * range_size, delta * range_size)

        return self.clip(result)

    def gradient_step(
        self, solution: list[float], gradient: list[float], step_size: float
    ) -> list[float]:
        """Take a gradient descent step.

        Args:
            solution: Current solution
            gradient: Gradient vector (same length as solution)
            step_size: Learning rate

        Returns:
            New solution after gradient step (clipped)
        """
        result = [x - step_size * g for x, g in zip(solution, gradient)]
        return self.clip(result)

    def gradient(self, solution: list[float], epsilon: float = 1e-6) -> list[float]:
        """Estimate gradient numerically (finite differences).

        Override in domain contexts if analytical gradient available.

        Args:
            solution: Point to evaluate gradient
            epsilon: Step size for finite differences

        Returns:
            Approximate gradient vector
        """
        grad = []
        base_cost = self.evaluate(solution)

        for i in range(self.dimension):
            perturbed = solution.copy()
            lower, upper = self.bounds[i]
            step = epsilon * (upper - lower)
            perturbed[i] = min(upper, solution[i] + step)
            grad.append((self.evaluate(perturbed) - base_cost) / step)

        return grad

    def crossover_blend(
        self, parent1: list[float], parent2: list[float], alpha: float = 0.5
    ) -> list[float]:
        """BLX-alpha crossover.

        Creates offspring in extended range between parents.

        Args:
            parent1: First parent solution
            parent2: Second parent solution
            alpha: Extension parameter (0.5 is common)

        Returns:
            Offspring solution (clipped to bounds)
        """
        result = []
        for i, (x1, x2) in enumerate(zip(parent1, parent2)):
            lower_p, upper_p = min(x1, x2), max(x1, x2)
            range_p = upper_p - lower_p
            lower_ext = lower_p - alpha * range_p
            upper_ext = upper_p + alpha * range_p
            result.append(random.uniform(lower_ext, upper_ext))

        return self.clip(result)

    def crossover_arithmetic(
        self, parent1: list[float], parent2: list[float], weight: float | None = None
    ) -> list[float]:
        """Arithmetic (weighted average) crossover.

        Args:
            parent1: First parent
            parent2: Second parent
            weight: Weight for parent1 (random if None)

        Returns:
            Offspring as weighted average
        """
        if weight is None:
            weight = random.random()

        result = [weight * x1 + (1 - weight) * x2 for x1, x2 in zip(parent1, parent2)]
        return self.clip(result)

    def differential_mutation(
        self,
        base: list[float],
        diff1: list[float],
        diff2: list[float],
        F: float = 0.8,
    ) -> list[float]:
        """Differential Evolution mutation.

        Creates mutant as: base + F * (diff1 - diff2)

        Args:
            base: Base vector
            diff1: First difference vector
            diff2: Second difference vector
            F: Scaling factor

        Returns:
            Mutant vector (clipped)
        """
        result = [
            b + F * (d1 - d2) for b, d1, d2 in zip(base, diff1, diff2)
        ]
        return self.clip(result)

    def apply_move(
        self, solution: list[float], move: str, i: int, j: int
    ) -> list[float] | None:
        """Apply a continuous move.

        Args:
            solution: Current solution
            move: "perturb" (uses i as index, j ignored)
            i: Index to modify
            j: Unused

        Returns:
            Modified solution
        """
        if move == "perturb":
            return self.perturb(solution, sigma=0.1, indices=[i])
        return None

    # =========================================================================
    # Distance metrics
    # =========================================================================

    def euclidean_distance(self, sol1: list[float], sol2: list[float]) -> float:
        """Euclidean distance between solutions.

        Args:
            sol1: First solution
            sol2: Second solution

        Returns:
            L2 distance
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(sol1, sol2)))

    def normalized_distance(self, sol1: list[float], sol2: list[float]) -> float:
        """Euclidean distance normalized by bounds.

        Args:
            sol1: First solution
            sol2: Second solution

        Returns:
            Normalized distance in [0, sqrt(n)]
        """
        total = 0.0
        for (x1, x2, (lower, upper)) in zip(sol1, sol2, self.bounds):
            range_size = upper - lower
            if range_size > 0:
                total += ((x1 - x2) / range_size) ** 2
        return math.sqrt(total)

    # =========================================================================
    # Universal methods (required by OptimizationContext)
    # =========================================================================

    def random_solution(self) -> list[float]:
        """Generate a random solution within bounds."""
        return [random.uniform(lower, upper) for lower, upper in self.bounds]

    def copy(self, solution: list[float]) -> list[float]:
        """Deep copy of solution."""
        return solution.copy()

    def valid(self, solution: list[float]) -> bool:
        """Check if solution is valid (within bounds)."""
        if len(solution) != self.dimension:
            return False
        return self.in_bounds(solution)

    # =========================================================================
    # Abstract methods (must be implemented by domain contexts)
    # =========================================================================

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'function_opt', 'hyperparameter')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of continuous variables."""

    @abstractmethod
    def evaluate(self, solution: list[float]) -> float:
        """Evaluate solution cost (domain-specific)."""


__all__ = ["ContinuousContext"]
