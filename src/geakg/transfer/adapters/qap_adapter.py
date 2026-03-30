"""QAP Adapter: Transfer TSP operators to QAP domain.

QAP (QAPLIB convention): assignment[i] = facility at location i
TSP: tour[i] = city at position i

Ambos son permutaciones de [0, 1, ..., n-1].

IMPORTANTE: QAP tiene semántica diferente a TSP:
- TSP: costo = suma de aristas (vecinos en tour)
- QAP: costo = Σ f[π(i)][π(j)] × d[i][j] (todas las parejas)

Por esto, ctx.delta() debe calcular el cambio REAL en QAP, no en TSP.
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


class QAPTransferContext:
    """Context optimizado para QAP con delta correcto.

    A diferencia de TransferContext (TSP), este calcula:
    - delta() usando la fórmula real de QAP
    - cost() como contribución global, no local
    """

    def __init__(self, flow_matrix: list, distance_matrix: list, n: int):
        self.flow_matrix = flow_matrix
        self.dist_matrix = distance_matrix  # QAP distance matrix
        self._n = n
        # Pseudo distance matrix for compatibility (not used for delta)
        self._pseudo_dm = [[0.0] * n for _ in range(n)]

    @property
    def distance_matrix(self):
        """For compatibility with operators that access this directly."""
        return self._pseudo_dm

    @distance_matrix.setter
    def distance_matrix(self, value):
        self._pseudo_dm = value

    def evaluate(self, assignment: list[int]) -> float:
        """Evaluate REAL QAP cost: Σ f[π(i)][π(j)] × d[i][j]."""
        total = 0
        n = len(assignment)
        for i in range(n):
            for j in range(n):
                total += self.flow_matrix[assignment[i]][assignment[j]] * self.dist_matrix[i][j]
        return float(total)

    def delta(self, assignment: list[int], move_type: str, i: int, j: int) -> float:
        """Calculate REAL delta for QAP swap.

        When swapping positions i and j in assignment:
        - Old: f[π(i)][π(k)] × d[i][k] + f[π(j)][π(k)] × d[j][k] for all k
        - New: f[π(j)][π(k)] × d[i][k] + f[π(i)][π(k)] × d[j][k] for all k

        Delta = new - old (only affected terms)
        """
        if move_type != "swap" or i == j:
            return 0.0

        n = len(assignment)
        pi_i = assignment[i]
        pi_j = assignment[j]

        delta = 0.0

        for k in range(n):
            if k == i or k == j:
                continue

            pi_k = assignment[k]

            # Terms involving position i
            # Old: f[π(i)][π(k)] × d[i][k] + f[π(k)][π(i)] × d[k][i]
            # New: f[π(j)][π(k)] × d[i][k] + f[π(k)][π(j)] × d[k][i]
            old_i = (self.flow_matrix[pi_i][pi_k] * self.distance_matrix[i][k] +
                     self.flow_matrix[pi_k][pi_i] * self.dist_matrix[k][i])
            new_i = (self.flow_matrix[pi_j][pi_k] * self.distance_matrix[i][k] +
                     self.flow_matrix[pi_k][pi_j] * self.dist_matrix[k][i])

            # Terms involving position j
            # Old: f[π(j)][π(k)] × d[j][k] + f[π(k)][π(j)] × d[k][j]
            # New: f[π(i)][π(k)] × d[j][k] + f[π(k)][π(i)] × d[k][j]
            old_j = (self.flow_matrix[pi_j][pi_k] * self.dist_matrix[j][k] +
                     self.flow_matrix[pi_k][pi_j] * self.dist_matrix[k][j])
            new_j = (self.flow_matrix[pi_i][pi_k] * self.dist_matrix[j][k] +
                     self.flow_matrix[pi_k][pi_i] * self.dist_matrix[k][j])

            delta += (new_i - old_i) + (new_j - old_j)

        # Terms between i and j themselves
        # Old: f[π(i)][π(j)] × d[i][j] + f[π(j)][π(i)] × d[j][i]
        # New: f[π(j)][π(i)] × d[i][j] + f[π(i)][π(j)] × d[j][i]
        old_ij = (self.flow_matrix[pi_i][pi_j] * self.dist_matrix[i][j] +
                  self.flow_matrix[pi_j][pi_i] * self.dist_matrix[j][i])
        new_ij = (self.flow_matrix[pi_j][pi_i] * self.dist_matrix[i][j] +
                  self.flow_matrix[pi_i][pi_j] * self.dist_matrix[j][i])
        delta += new_ij - old_ij

        return delta

    def cost(self, assignment: list[int], i: int) -> float:
        """Cost contribution of element at position i.

        For QAP: sum of all interactions involving position i.
        """
        n = len(assignment)
        pi_i = assignment[i]
        total = 0.0

        for k in range(n):
            if k != i:
                pi_k = assignment[k]
                total += self.flow_matrix[pi_i][pi_k] * self.distance_matrix[i][k]
                total += self.flow_matrix[pi_k][pi_i] * self.dist_matrix[k][i]

        return total

    def neighbors(self, assignment: list[int], i: int, k: int) -> list[int]:
        """Returns k positions with highest interaction with position i."""
        n = len(assignment)
        pi_i = assignment[i]

        interactions = []
        for j in range(n):
            if j != i:
                pi_j = assignment[j]
                # Total interaction between positions i and j
                interaction = (self.flow_matrix[pi_i][pi_j] * self.dist_matrix[i][j] +
                              self.flow_matrix[pi_j][pi_i] * self.dist_matrix[j][i])
                interactions.append((interaction, j))

        # Sort by interaction (highest first - these are candidates for swap)
        interactions.sort(reverse=True)
        return [idx for _, idx in interactions[:k]]

    def valid(self, assignment: list[int]) -> bool:
        """Check if valid permutation."""
        if not assignment:
            return False
        n = len(assignment)
        return len(set(assignment)) == n and all(0 <= x < self._n for x in assignment)


@dataclass
class QAPAssignment:
    """QAP solution representation."""

    assignment: list[int]  # assignment[i] = facility at location i (QAPLIB convention)
    cost: int = 0


class QAPAdapter(DomainAdapter[QAPAssignment, Any]):
    """Adapter for transferring TSP operators to QAP.

    Mapping is trivial (both are permutations).
    Distance matrix combines flow and distance information.
    """

    def __init__(
        self,
        qap_instance: Any,
        config: AdapterConfig | None = None,
    ):
        super().__init__(qap_instance, config)
        self.n = qap_instance.n

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "qap"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from QAP instance.

        Distance matrix heuristic: combine flow and distance info.
        d_tsp[i][j] = f[i][j] × avg_dist (interaction strength)
        """
        instance = self.target_instance
        n = instance.n

        # Calculate average distance
        total_dist = sum(sum(row) for row in instance.distance_matrix)
        avg_dist = total_dist / (n * n) if n > 0 else 1

        # Create TSP distance matrix based on flow
        # High flow between i,j means they should be "close" in TSP terms
        # So use negative flow (or max_flow - flow) as distance
        max_flow = max(max(row) for row in instance.flow_matrix) + 1

        tsp_dm = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Lower flow = higher distance
                    tsp_dm[i][j] = float(max_flow - instance.flow_matrix[i][j])

        return SourceContext(
            distance_matrix=tsp_dm,
            index_to_element={i: i for i in range(n)},
            element_to_index={i: i for i in range(n)},
            extra={
                "flow_matrix": instance.flow_matrix,
                "distance_matrix": instance.distance_matrix,
                "n": n,
            },
        )

    def to_source_repr(
        self,
        qap_solution: QAPAssignment | list[int],
    ) -> tuple[list[int], SourceContext]:
        """Convert QAP assignment to TSP tour. Trivial mapping."""
        # Handle different solution types
        if hasattr(qap_solution, 'assignment'):
            # QAPAssignment or QAPSolution (Pydantic)
            assignment = qap_solution.assignment
        elif isinstance(qap_solution, list):
            assignment = qap_solution
        else:
            raise ValueError(f"Unknown solution type: {type(qap_solution)}")

        return list(assignment), self.get_context()

    def from_source_repr(
        self,
        tsp_tour: list[int],
        context: SourceContext | None = None,
    ) -> QAPAssignment:
        """Convert TSP tour to QAP assignment. Trivial mapping."""
        assignment = list(tsp_tour)
        cost = self._compute_cost(assignment)
        return QAPAssignment(assignment=assignment, cost=cost)

    def _compute_cost(self, assignment: list[int]) -> int:
        """Compute QAP cost: Σ f_π(i),π(j) × d_i,j.

        QAPLIB convention: assignment[i] = facility at location i.
        """
        instance = self.target_instance
        n = instance.n
        total = 0
        for i in range(n):
            for j in range(n):
                flow = instance.flow_matrix[assignment[i]][assignment[j]]
                dist = instance.distance_matrix[i][j]
                total += flow * dist
        return total

    def get_evaluate_fn(self) -> callable:
        """Return QAP evaluation function for use in operators."""
        def evaluate_qap(assignment: list[int]) -> float:
            return float(self._compute_cost(assignment))
        return evaluate_qap

    def get_qap_context(self) -> QAPTransferContext:
        """Get QAP-specific context with correct delta calculation."""
        instance = self.target_instance
        return QAPTransferContext(
            flow_matrix=instance.flow_matrix,
            distance_matrix=instance.distance_matrix,
            n=instance.n,
        )

    def adapt_operator(
        self,
        operator_id: str,
        operator_fn: callable,
        role: "AbstractRole",
        original_code: str = "",
    ) -> "AdaptedOperator":
        """Adapt operator using QAP-specific context.

        Override base class to use QAPTransferContext which has
        correct delta() and cost() for QAP semantics.
        """
        from src.geakg.transfer.adapter import AdaptedOperator

        qap_ctx = self.get_qap_context()

        def adapted_fn(
            target_solution: QAPAssignment,
            target_instance: Any = None,
        ) -> QAPAssignment:
            """Wrapped operator using QAP context."""
            # Get assignment
            if hasattr(target_solution, 'assignment'):
                assignment = list(target_solution.assignment)
            else:
                assignment = list(target_solution)

            # Apply operator with QAP context
            try:
                improved = operator_fn(assignment, qap_ctx)
            except Exception:
                return target_solution

            # Convert to QAPAssignment
            if isinstance(improved, list):
                cost = self._compute_cost(improved)
                result = QAPAssignment(assignment=improved, cost=cost)
            else:
                result = improved

            # Validate
            if self.validate_result(result):
                return result

            # Repair if needed
            if self.config.repair_violations:
                repaired = self._repair_solution(result)
                if self.validate_result(repaired):
                    return repaired

            return target_solution

        return AdaptedOperator(
            operator_id=f"{operator_id}_{self.target_domain}",
            original_id=operator_id,
            role=role,
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            weight=1.0,
            description=f"Adapted from {self.source_domain}: {operator_id}",
            adapted_fn=adapted_fn,
            original_code=original_code,
        )

    def validate_result(self, result: QAPAssignment) -> bool:
        """Validate is valid permutation."""
        assignment = result.assignment
        if len(assignment) != self.n:
            return False
        if set(assignment) != set(range(self.n)):
            return False
        return True

    def _repair_solution(self, solution: QAPAssignment) -> QAPAssignment:
        """Repair invalid permutation."""
        assignment = solution.assignment
        seen = set()
        repaired = []
        for x in assignment:
            if 0 <= x < self.n and x not in seen:
                repaired.append(x)
                seen.add(x)
        for x in range(self.n):
            if x not in seen:
                repaired.append(x)
        cost = self._compute_cost(repaired)
        return QAPAssignment(assignment=repaired, cost=cost)


def create_qap_adapter(qap_instance: Any) -> QAPAdapter:
    """Factory function to create QAP adapter."""
    config = AdapterConfig(source_domain="tsp", repair_violations=True)
    return QAPAdapter(qap_instance, config)
