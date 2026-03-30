"""LOP Adapter: Transfer TSP operators to LOP domain.

LOP es el adaptador más trivial:
- TSP: permutación [0, 1, ..., n-1], minimizar distancia
- LOP: permutación [0, 1, ..., n-1], maximizar suma

La única diferencia es que LOP maximiza y TSP minimiza.
Usamos la matriz C directamente (negada) como "distancias".
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


@dataclass
class LOPPermutation:
    """LOP solution representation."""

    permutation: list[int]
    value: int = 0  # Objective value (to maximize)

    @property
    def cost(self) -> float:
        """Negative value for minimization framework."""
        return float(-self.value)


class LOPAdapter(DomainAdapter[LOPPermutation, Any]):
    """Adapter for transferring TSP operators to LOP.

    Mapping is trivial:
    - TSP tour [0, 1, ..., n-1] = LOP permutation [0, 1, ..., n-1]

    Distance matrix is the negated LOP matrix (to convert max to min).
    """

    def __init__(
        self,
        lop_instance: Any,
        config: AdapterConfig | None = None,
    ):
        super().__init__(lop_instance, config)
        self.n = lop_instance.n

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "lop"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from LOP instance.

        Use negated matrix as distances (LOP maximizes, TSP minimizes).
        """
        instance = self.target_instance
        n = instance.n

        # Negate matrix for minimization
        # Also add offset to make all values positive
        max_val = max(max(row) for row in instance.matrix)
        tsp_dm = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                # Higher LOP value = lower "distance"
                tsp_dm[i][j] = float(max_val - instance.matrix[i][j] + 1)

        return SourceContext(
            distance_matrix=tsp_dm,
            index_to_element={i: i for i in range(n)},
            element_to_index={i: i for i in range(n)},
            extra={"matrix": instance.matrix, "n": n},
        )

    def to_source_repr(
        self,
        lop_solution: LOPPermutation | list[int],
    ) -> tuple[list[int], SourceContext]:
        """Convert LOP permutation to TSP tour. Trivial mapping."""
        if isinstance(lop_solution, LOPPermutation):
            perm = lop_solution.permutation
        else:
            perm = lop_solution

        return list(perm), self.get_context()

    def from_source_repr(
        self,
        tsp_tour: list[int],
        context: SourceContext | None = None,
    ) -> LOPPermutation:
        """Convert TSP tour to LOP permutation. Trivial mapping."""
        perm = list(tsp_tour)
        value = self._compute_value(perm)
        return LOPPermutation(permutation=perm, value=value)

    def _compute_value(self, perm: list[int]) -> int:
        """Compute LOP objective value."""
        matrix = self.target_instance.matrix
        n = len(perm)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += matrix[perm[i]][perm[j]]
        return total

    def validate_result(self, result: LOPPermutation) -> bool:
        """Validate is valid permutation."""
        perm = result.permutation
        if len(perm) != self.n:
            return False
        if set(perm) != set(range(self.n)):
            return False
        return True

    def _repair_solution(self, solution: LOPPermutation) -> LOPPermutation:
        """Repair invalid permutation."""
        perm = solution.permutation
        seen = set()
        repaired = []
        for x in perm:
            if 0 <= x < self.n and x not in seen:
                repaired.append(x)
                seen.add(x)
        for x in range(self.n):
            if x not in seen:
                repaired.append(x)
        value = self._compute_value(repaired)
        return LOPPermutation(permutation=repaired, value=value)


def create_lop_adapter(lop_instance: Any) -> LOPAdapter:
    """Factory function to create LOP adapter."""
    config = AdapterConfig(source_domain="tsp", repair_violations=True)
    return LOPAdapter(lop_instance, config)
