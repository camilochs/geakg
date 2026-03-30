"""Linear Ordering Problem (LOP) domain implementation.

LOP: Dada una matriz C de n×n, encontrar una permutación π que
maximice la suma de elementos "arriba de la diagonal":
    Σ c_π(i),π(j) para todo i < j

Representación: Permutación [0, 1, ..., n-1]
Objetivo: Maximizar (o minimizar negativo)
"""

import random
from pathlib import Path

from pydantic import BaseModel, Field


class LOPInstance(BaseModel):
    """LOP problem instance.

    Attributes:
        name: Instance identifier
        n: Matrix dimension
        matrix: Weight matrix C[i][j]
        optimal_value: Known optimal (if available)
    """

    name: str
    n: int = Field(gt=0)
    matrix: list[list[int]]
    optimal_value: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class LOPSolution(BaseModel):
    """LOP solution.

    Attributes:
        permutation: Order of elements [0, 1, ..., n-1]
        value: Objective value (to maximize)
    """

    permutation: list[int]
    value: int = 0

    @property
    def cost(self) -> float:
        """Return negative value (we minimize in transfer framework)."""
        return float(-self.value)


class LOPDomain:
    """LOP domain implementation."""

    @property
    def name(self) -> str:
        return "lop"

    def load_instance(self, path: Path) -> LOPInstance:
        """Load LOP instance from file.

        Format:
            n
            matrix (n rows, n columns)
        """
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        # First non-comment line is n
        n = int(lines[0])

        # Read matrix
        matrix = []
        for i in range(1, n + 1):
            row = list(map(int, lines[i].split()))
            matrix.append(row)

        # Try to get optimal from comments
        optimal = None
        with open(path) as f:
            for line in f:
                if "optimal" in line.lower():
                    try:
                        optimal = int(line.split()[-1])
                    except ValueError:
                        pass

        return LOPInstance(
            name=path.stem,
            n=n,
            matrix=matrix,
            optimal_value=optimal,
        )

    def evaluate_solution(self, solution: LOPSolution, instance: LOPInstance) -> float:
        """Compute objective value (negated for minimization framework).

        LOP maximizes sum above diagonal. We return negative for minimization.
        """
        value = self._compute_value(solution.permutation, instance)
        return float(-value)  # Negate for minimization

    def _compute_value(self, perm: list[int], instance: LOPInstance) -> int:
        """Compute LOP objective value.

        Sum of matrix[perm[i]][perm[j]] for all i < j.
        """
        n = len(perm)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += instance.matrix[perm[i]][perm[j]]
        return total

    def validate_solution(self, solution: LOPSolution, instance: LOPInstance) -> bool:
        """Check if solution is a valid permutation."""
        perm = solution.permutation
        if len(perm) != instance.n:
            return False
        if set(perm) != set(range(instance.n)):
            return False
        return True

    def random_solution(self, instance: LOPInstance) -> LOPSolution:
        """Generate random permutation."""
        perm = list(range(instance.n))
        random.shuffle(perm)
        value = self._compute_value(perm, instance)
        return LOPSolution(permutation=perm, value=value)

    def greedy_solution(self, instance: LOPInstance) -> LOPSolution:
        """Greedy construction: insert element that maximizes contribution."""
        n = instance.n
        remaining = set(range(n))
        perm = []

        # Start with element that has highest row sum
        row_sums = [sum(instance.matrix[i]) for i in range(n)]
        first = max(remaining, key=lambda x: row_sums[x])
        perm.append(first)
        remaining.remove(first)

        while remaining:
            best_elem = None
            best_pos = 0
            best_gain = float("-inf")

            for elem in remaining:
                # Try inserting at each position
                for pos in range(len(perm) + 1):
                    gain = self._insertion_gain(perm, pos, elem, instance)
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = elem
                        best_pos = pos

            perm = perm[:best_pos] + [best_elem] + perm[best_pos:]
            remaining.remove(best_elem)

        value = self._compute_value(perm, instance)
        return LOPSolution(permutation=perm, value=value)

    def _insertion_gain(
        self, perm: list[int], pos: int, elem: int, instance: LOPInstance
    ) -> int:
        """Calculate gain from inserting elem at position pos."""
        gain = 0
        # Elements before pos: elem is after them
        for i in range(pos):
            gain += instance.matrix[perm[i]][elem]
        # Elements after pos: elem is before them
        for i in range(pos, len(perm)):
            gain += instance.matrix[elem][perm[i]]
        return gain


    def becker_solution(self, instance: LOPInstance) -> LOPSolution:
        """Becker's heuristic for LOP (1967).

        Classic LOP-specific heuristic: order elements by
        (row_sum - col_sum) in decreasing order.

        Elements with high row_sum and low col_sum should come first
        (they "dominate" others).
        """
        n = instance.n

        # Calculate row_sum - col_sum for each element
        scores = []
        for i in range(n):
            row_sum = sum(instance.matrix[i])
            col_sum = sum(instance.matrix[j][i] for j in range(n))
            scores.append((row_sum - col_sum, i))

        # Sort by decreasing score
        scores.sort(reverse=True)
        perm = [i for _, i in scores]

        value = self._compute_value(perm, instance)
        return LOPSolution(permutation=perm, value=value)


def create_sample_lop_instance(n: int = 10, seed: int = 42) -> LOPInstance:
    """Create a sample LOP instance for testing."""
    random.seed(seed)

    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = random.randint(0, 100)

    return LOPInstance(
        name=f"sample_{n}",
        n=n,
        matrix=matrix,
    )
