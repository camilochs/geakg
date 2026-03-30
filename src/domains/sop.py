"""Sequential Ordering Problem (SOP) domain implementation.

SOP: Find the shortest Hamiltonian path from node 1 to node n while
respecting precedence constraints (node i must be visited before node j).

It's essentially TSP with precedence constraints.
"""

import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.domains.base import OptimizationDomain, ProblemFeatures


class SOPInstance(BaseModel):
    """SOP problem instance.

    Attributes:
        name: Instance identifier
        n: Number of nodes
        distance_matrix: Distance matrix (n x n)
        precedences: List of (i, j) pairs where i must precede j
    """

    name: str
    n: int = Field(gt=0)
    distance_matrix: list[list[float]]
    precedences: list[tuple[int, int]] = Field(default_factory=list)
    optimal_length: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class SOPSolution(BaseModel):
    """SOP solution (Hamiltonian path).

    Attributes:
        path: Sequence of nodes from start to end
        length: Total path length
    """

    path: list[int]
    length: float = 0.0
    is_valid: bool = True

    @property
    def cost(self) -> float:
        """Alias for length as cost."""
        return float(self.length)


class SOPFeatures(ProblemFeatures):
    """Features extracted from SOP instance."""

    dimension: int
    n_precedences: int
    avg_distance: float
    precedence_density: float  # n_precedences / (n * (n-1))

    @classmethod
    def from_instance(cls, instance: SOPInstance) -> "SOPFeatures":
        """Extract features from SOP instance."""
        n = instance.n
        all_dists = [
            instance.distance_matrix[i][j]
            for i in range(n)
            for j in range(n)
            if i != j and instance.distance_matrix[i][j] < float("inf")
        ]

        avg_dist = sum(all_dists) / len(all_dists) if all_dists else 0
        max_precedences = n * (n - 1)
        density = len(instance.precedences) / max_precedences if max_precedences > 0 else 0

        return cls(
            dimension=n,
            n_precedences=len(instance.precedences),
            avg_distance=avg_dist,
            precedence_density=density,
        )


class SOPDomain(OptimizationDomain[SOPInstance, SOPSolution]):
    """SOP domain implementation."""

    @property
    def name(self) -> str:
        return "sop"

    def load_instance(self, path: Path) -> SOPInstance:
        """Load SOP instance from TSPLIB format.

        Args:
            path: Path to instance file

        Returns:
            Loaded SOP instance
        """
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        name = path.stem
        n = 0
        distance_matrix = []
        precedences = []

        # Parse TSPLIB format
        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                n = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                # Read distance matrix
                i += 1
                # Skip dimension line if present (some SOP files have it)
                if i < len(lines) and lines[i].strip().isdigit():
                    i += 1
                distance_matrix = []
                for _ in range(n):
                    if i >= len(lines):
                        break
                    row = list(map(float, lines[i].split()))
                    distance_matrix.append(row)
                    i += 1
                i -= 1  # Adjust because we'll increment at loop end
            elif line.startswith("PRECEDENCE_SECTION"):
                # Read precedence constraints
                i += 1
                while i < len(lines) and not lines[i].startswith("-1"):
                    parts = list(map(int, lines[i].split()))
                    if len(parts) >= 2:
                        # Format: node followed by list of successors, terminated by -1
                        node = parts[0]
                        successors = [s for s in parts[1:] if s != -1]
                        for succ in successors:
                            precedences.append((node - 1, succ - 1))  # Convert to 0-indexed
                    i += 1
                i -= 1

            i += 1

        # If no explicit precedences, infer from -1 entries in distance matrix
        # In SOP TSPLIB format, d[i][j] = -1 means:
        #   - Cannot go directly from i to j
        #   - This implies j must be visited BEFORE i (precedence: j precedes i)
        if not precedences and distance_matrix:
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix[i])):
                    if distance_matrix[i][j] == -1:
                        # j must come before i (since we can't go from i to j)
                        precedences.append((j, i))
                        # Convert -1 to infinity for pathfinding
                        distance_matrix[i][j] = float('inf')

        return SOPInstance(
            name=name,
            n=n,
            distance_matrix=distance_matrix,
            precedences=precedences,
        )

    def evaluate_solution(self, solution: SOPSolution, instance: SOPInstance) -> float:
        """Compute path length.

        Args:
            solution: SOP solution (path)
            instance: SOP instance

        Returns:
            Path length (lower is better)
        """
        length = self._compute_path_length(solution.path, instance)
        return float(length)

    def _compute_path_length(self, path: list[int], instance: SOPInstance) -> float:
        """Compute total path length.

        Args:
            path: Sequence of nodes
            instance: SOP instance

        Returns:
            Total distance
        """
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            total += instance.distance_matrix[from_node][to_node]

        return total

    def validate_solution(self, solution: SOPSolution, instance: SOPInstance) -> bool:
        """Check if path is valid (visits all nodes and respects precedences).

        Args:
            solution: SOP solution
            instance: SOP instance

        Returns:
            True if valid path
        """
        path = solution.path

        # Check all nodes visited exactly once
        if len(path) != instance.n or len(set(path)) != instance.n:
            return False

        # Check all nodes are valid
        if any(node < 0 or node >= instance.n for node in path):
            return False

        # Check precedence constraints
        position = {node: i for i, node in enumerate(path)}
        for before, after in instance.precedences:
            if position[before] >= position[after]:
                return False

        return True

    def get_features(self, instance: SOPInstance) -> SOPFeatures:
        """Extract features from SOP instance.

        Args:
            instance: SOP instance

        Returns:
            Extracted features
        """
        return SOPFeatures.from_instance(instance)

    def random_solution(self, instance: SOPInstance) -> SOPSolution:
        """Generate a random valid path respecting precedences.

        Uses topological sort with random choices.

        Args:
            instance: SOP instance

        Returns:
            Random valid path
        """
        n = instance.n

        # Build precedence graph
        successors = {i: set() for i in range(n)}
        predecessors = {i: set() for i in range(n)}

        for before, after in instance.precedences:
            successors[before].add(after)
            predecessors[after].add(before)

        # Topological sort with random selection
        available = [i for i in range(n) if not predecessors[i]]
        path = []

        while available:
            # Random choice among available nodes
            node = random.choice(available)
            path.append(node)
            available.remove(node)

            # Update available nodes
            for succ in successors[node]:
                predecessors[succ].discard(node)
                if not predecessors[succ]:
                    available.append(succ)

        solution = SOPSolution(path=path)
        solution.length = self._compute_path_length(path, instance)
        solution.is_valid = self.validate_solution(solution, instance)

        return solution

    def greedy_solution(self, instance: SOPInstance) -> SOPSolution:
        """Generate solution using greedy nearest neighbor respecting precedences.

        Args:
            instance: SOP instance

        Returns:
            Greedy path
        """
        n = instance.n

        # Build precedence constraints
        successors = {i: set() for i in range(n)}
        predecessors = {i: set() for i in range(n)}

        for before, after in instance.precedences:
            successors[before].add(after)
            predecessors[after].add(before)

        # Start from nodes with no predecessors
        available = [i for i in range(n) if not predecessors[i]]
        if not available:
            # Fallback to random if no valid start
            return self.random_solution(instance)

        # Start with closest to first available
        current = min(available, key=lambda i: instance.distance_matrix[0][i] if i != 0 else float("inf"))
        path = [current]
        available.remove(current)

        # Update available after selecting first
        for succ in successors[current]:
            predecessors[succ].discard(current)
            if not predecessors[succ]:
                available.append(succ)

        # Greedy: always pick nearest available node
        while available:
            # Find nearest available node
            nearest = min(available, key=lambda i: instance.distance_matrix[current][i])
            path.append(nearest)
            current = nearest
            available.remove(nearest)

            # Update available
            for succ in successors[nearest]:
                predecessors[succ].discard(nearest)
                if not predecessors[succ]:
                    available.append(succ)

        solution = SOPSolution(path=path)
        solution.length = self._compute_path_length(path, instance)
        solution.is_valid = self.validate_solution(solution, instance)

        return solution


def create_sample_sop_instance(n: int = 10, seed: int = 42) -> SOPInstance:
    """Create a sample SOP instance for testing.

    Args:
        n: Number of nodes
        seed: Random seed

    Returns:
        Sample SOP instance
    """
    random.seed(seed)

    # Random distance matrix
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = random.randint(1, 100)

    # Random precedence constraints (sparse)
    precedences = []
    for _ in range(n // 2):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        precedences.append((i, j))

    return SOPInstance(
        name=f"sample_sop_{n}",
        n=n,
        distance_matrix=distance_matrix,
        precedences=precedences,
    )
