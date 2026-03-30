"""Quadratic Assignment Problem (QAP) domain implementation.

QAP: Asignar n facilidades a n locaciones minimizando:
    Σ f_π(i),π(j) × d_i,j
donde f es matriz de flujos, d es matriz de distancias,
y π(i) es la facilidad asignada a locación i.

Representación (convención QAPLIB): π[i] = facilidad en locación i
Objetivo: Minimizar costo total
"""

import random
from pathlib import Path

from pydantic import BaseModel, Field


class QAPInstance(BaseModel):
    """QAP problem instance.

    Attributes:
        name: Instance identifier
        n: Number of facilities/locations
        flow_matrix: f[i][j] = flow between facility i and j
        distance_matrix: d[i][j] = distance between location i and j
        optimal_cost: Known optimal (if available)
    """

    name: str
    n: int = Field(gt=0)
    flow_matrix: list[list[int]]
    distance_matrix: list[list[int]]
    optimal_cost: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class QAPSolution(BaseModel):
    """QAP solution.

    Attributes:
        assignment: assignment[i] = facility at location i (QAPLIB convention)
        cost: Total cost
    """

    assignment: list[int]
    cost: int = 0


class QAPDomain:
    """QAP domain implementation."""

    @property
    def name(self) -> str:
        return "qap"

    def load_instance(self, path: Path) -> QAPInstance:
        """Load QAP instance from QAPLIB format.

        Format:
            n
            (blank line)
            distance_matrix (n rows)
            (blank line)
            flow_matrix (n rows)
        """
        with open(path) as f:
            content = f.read()

        # Remove comments and parse
        lines = []
        optimal = None
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                if "optimal" in line.lower():
                    try:
                        optimal = int(line.split()[-1])
                    except ValueError:
                        pass
                continue
            if line.strip():
                lines.append(line.strip())

        n = int(lines[0])

        # Parse distance matrix (first n rows after n)
        distance_matrix = []
        idx = 1
        while len(distance_matrix) < n:
            row = list(map(int, lines[idx].split()))
            distance_matrix.append(row)
            idx += 1

        # Parse flow matrix (next n rows)
        flow_matrix = []
        while len(flow_matrix) < n:
            row = list(map(int, lines[idx].split()))
            flow_matrix.append(row)
            idx += 1

        return QAPInstance(
            name=path.stem,
            n=n,
            flow_matrix=flow_matrix,
            distance_matrix=distance_matrix,
            optimal_cost=optimal,
        )

    def evaluate_solution(self, solution: QAPSolution, instance: QAPInstance) -> float:
        """Compute QAP cost."""
        cost = self._compute_cost(solution.assignment, instance)
        return float(cost)

    def _compute_cost(self, assignment: list[int], instance: QAPInstance) -> int:
        """Compute QAP objective: Σ f_π(i),π(j) × d_i,j.

        QAPLIB convention: assignment[i] = facility at location i.
        Cost = sum over all location pairs (i,j) of:
               flow between facilities at those locations × distance between locations
        """
        n = instance.n
        total = 0
        for i in range(n):
            for j in range(n):
                # assignment[i] = facility at location i
                # assignment[j] = facility at location j
                flow = instance.flow_matrix[assignment[i]][assignment[j]]
                dist = instance.distance_matrix[i][j]
                total += flow * dist
        return total

    def validate_solution(self, solution: QAPSolution, instance: QAPInstance) -> bool:
        """Check if assignment is a valid permutation."""
        assignment = solution.assignment
        if len(assignment) != instance.n:
            return False
        if set(assignment) != set(range(instance.n)):
            return False
        return True

    def random_solution(self, instance: QAPInstance) -> QAPSolution:
        """Generate random assignment."""
        assignment = list(range(instance.n))
        random.shuffle(assignment)
        cost = self._compute_cost(assignment, instance)
        return QAPSolution(assignment=assignment, cost=cost)

    def greedy_solution(self, instance: QAPInstance) -> QAPSolution:
        """Greedy: assign high-flow facilities to central locations."""
        n = instance.n

        # Calculate total flow for each facility
        flow_sums = [sum(instance.flow_matrix[i]) + sum(row[i] for row in instance.flow_matrix)
                     for i in range(n)]

        # Calculate centrality for each location (lower total distance = more central)
        dist_sums = [sum(instance.distance_matrix[i]) + sum(row[i] for row in instance.distance_matrix)
                     for i in range(n)]

        # Sort facilities by flow (descending) and locations by centrality (ascending)
        facilities = sorted(range(n), key=lambda x: -flow_sums[x])
        locations = sorted(range(n), key=lambda x: dist_sums[x])

        # Assign high-flow facilities to central locations
        assignment = [0] * n
        for i, facility in enumerate(facilities):
            assignment[facility] = locations[i]

        cost = self._compute_cost(assignment, instance)
        return QAPSolution(assignment=assignment, cost=cost)


    def gilmore_lawler_solution(self, instance: QAPInstance) -> QAPSolution:
        """Gilmore-Lawler heuristic for QAP (1962).

        Classic QAP-specific heuristic:
        1. Sort facilities by total flow (descending)
        2. Sort locations by total distance (ascending)
        3. Assign i-th facility to i-th location

        Simple but effective domain-specific heuristic.
        """
        n = instance.n

        # Total flow for each facility
        flow_totals = []
        for i in range(n):
            total = sum(instance.flow_matrix[i]) + sum(row[i] for row in instance.flow_matrix)
            flow_totals.append((total, i))

        # Total distance for each location
        dist_totals = []
        for i in range(n):
            total = sum(instance.distance_matrix[i]) + sum(row[i] for row in instance.distance_matrix)
            dist_totals.append((total, i))

        # Sort: facilities by decreasing flow, locations by increasing distance
        flow_totals.sort(reverse=True)
        dist_totals.sort()

        # Assign high-flow facilities to low-distance (central) locations
        assignment = [0] * n
        for rank, (_, facility) in enumerate(flow_totals):
            _, location = dist_totals[rank]
            assignment[facility] = location

        cost = self._compute_cost(assignment, instance)
        return QAPSolution(assignment=assignment, cost=cost)


def create_sample_qap_instance(n: int = 10, seed: int = 42) -> QAPInstance:
    """Create a sample QAP instance for testing."""
    random.seed(seed)

    flow_matrix = [[0] * n for _ in range(n)]
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Symmetric matrices
            flow = random.randint(0, 10)
            dist = random.randint(1, 10)
            flow_matrix[i][j] = flow_matrix[j][i] = flow
            distance_matrix[i][j] = distance_matrix[j][i] = dist

    return QAPInstance(
        name=f"sample_{n}",
        n=n,
        flow_matrix=flow_matrix,
        distance_matrix=distance_matrix,
    )
