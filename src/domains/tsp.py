"""Traveling Salesman Problem (TSP) domain implementation."""

import math
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.domains.base import DomainConfig


class TSPInstance(BaseModel):
    """TSP problem instance."""

    name: str
    dimension: int = Field(gt=0)
    coordinates: list[tuple[float, float]] | None = None
    distance_matrix: list[list[float]]
    edge_weight_type: str = "EUC_2D"
    optimal_cost: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class TSPDomain:
    """TSP domain - loads TSPLIB instances."""

    @property
    def name(self) -> str:
        return "tsp"

    def load_instance(self, path: Path) -> TSPInstance:
        """Load TSP instance from TSPLIB format."""
        with open(path) as f:
            content = f.read()
        return self._parse_tsplib(content, path.stem)

    def _parse_tsplib(self, content: str, name: str) -> TSPInstance:
        """Parse TSPLIB format content."""
        lines = content.strip().split("\n")

        dimension = 0
        edge_weight_type = "EUC_2D"
        edge_weight_format = None
        optimal_cost = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                i += 1
                break
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                i += 1
                break
            elif "optimal" in line.lower() or "best" in line.lower():
                # Try to match "optimal=N" or "optimal: N" or "optimal N" patterns
                match = re.search(r"optimal\s*[=:]\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
                if not match:
                    # Fallback: try "best known" or just a number after "optimal"
                    match = re.search(r"(?:optimal|best)\D*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
                if match:
                    optimal_cost = float(match.group(1))
            i += 1

        coordinates = None
        distance_matrix = None

        if edge_weight_type in ("EUC_2D", "CEIL_2D", "ATT", "GEO"):
            coordinates = []
            while i < len(lines):
                line = lines[i].strip()
                if line == "EOF" or line.startswith("DISPLAY"):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append((x, y))
                i += 1

            distance_matrix = self._compute_distance_matrix(coordinates, edge_weight_type)
            dimension = len(coordinates)

        elif edge_weight_type == "EXPLICIT":
            values = []
            while i < len(lines):
                line = lines[i].strip()
                if line == "EOF" or line.startswith("DISPLAY"):
                    break
                values.extend([float(x) for x in line.split() if x])
                i += 1

            distance_matrix = self._parse_explicit_matrix(values, dimension, edge_weight_format)

        if distance_matrix is None:
            raise ValueError(f"Could not parse distance matrix from {name}")

        return TSPInstance(
            name=name,
            dimension=dimension,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            edge_weight_type=edge_weight_type,
            optimal_cost=optimal_cost,
        )

    def _compute_distance_matrix(
        self, coordinates: list[tuple[float, float]], weight_type: str
    ) -> list[list[float]]:
        """Compute distance matrix from coordinates."""
        n = len(coordinates)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]

                    if weight_type == "EUC_2D":
                        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    elif weight_type == "CEIL_2D":
                        dist = math.ceil(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                    elif weight_type == "ATT":
                        xd, yd = x2 - x1, y2 - y1
                        rij = math.sqrt((xd * xd + yd * yd) / 10.0)
                        tij = round(rij)
                        dist = tij + 1 if tij < rij else tij
                    elif weight_type == "GEO":
                        dist = self._geo_distance(x1, y1, x2, y2)
                    else:
                        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    matrix[i][j] = dist

        return matrix

    def _geo_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute geographic distance (TSPLIB GEO format)."""
        PI = 3.141592
        RRR = 6378.388

        def to_radians(x: float) -> float:
            deg = int(x)
            min_ = x - deg
            return PI * (deg + 5.0 * min_ / 3.0) / 180.0

        lat1_r, lon1_r = to_radians(lat1), to_radians(lon1)
        lat2_r, lon2_r = to_radians(lat2), to_radians(lon2)

        q1 = math.cos(lon1_r - lon2_r)
        q2 = math.cos(lat1_r - lat2_r)
        q3 = math.cos(lat1_r + lat2_r)

        return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)

    def _parse_explicit_matrix(
        self, values: list[float], n: int, format_type: str | None
    ) -> list[list[float]]:
        """Parse explicit distance matrix from values."""
        matrix = [[0.0] * n for _ in range(n)]

        if format_type == "FULL_MATRIX":
            idx = 0
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = values[idx]
                    idx += 1
        elif format_type == "UPPER_ROW":
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j] = matrix[j][i] = values[idx]
                    idx += 1
        elif format_type == "LOWER_DIAG_ROW":
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    matrix[i][j] = matrix[j][i] = values[idx]
                    idx += 1
        elif format_type == "UPPER_DIAG_ROW":
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    matrix[i][j] = matrix[j][i] = values[idx]
                    idx += 1
        else:
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if idx < len(values):
                        matrix[i][j] = matrix[j][i] = values[idx]
                        idx += 1

        return matrix


# Known optimal values for TSPLIB instances
TSPLIB_OPTIMA = {
    "burma14": 3323,
    "ulysses22": 7013,
    "berlin52": 7542,
    "eil51": 426,
    "kroA100": 21282,
    "ch150": 6528,
    "tsp225": 3916,
    "a280": 2579,
    "pcb442": 50778,
    "rat783": 8806,
}


# =============================================================================
# SYNTHESIS CONFIGURATION
# =============================================================================

from src.prompts import SYNTHESIS_TEMPLATE

# TSP-specific configuration
TSP_DOMAIN_NAME = "TSP"

# AGNOSTIC signature - operators use ctx instead of distance_matrix
TSP_FUNCTION_SIGNATURE = "def operator_name(solution: list, ctx) -> list:"

# TSP-specific guidance for AGNOSTIC operators (OPTIMIZED: ~150 tokens saved)
TSP_DOMAIN_GUIDANCE = """
**CONTEXT API** (use ctx, NOT distance_matrix):
- `ctx.cost(sol, i)` → cost at position i
- `ctx.delta(sol, "swap", i, j)` → delta for swap (negative=better)
- `ctx.neighbors(sol, i, k)` → k nearest positions
- `ctx.evaluate(sol)` → total cost
- `ctx.valid(sol)` → True if valid

**RULES:** solution is a permutation [0..n-1]. Copy first (`result = solution[:]`). Use `(i+1) % n` for wrap-around. Validate before return.
"""


class TSPConfig(DomainConfig):
    """TSP configuration for operator synthesis with agnostic operators."""

    @property
    def name(self) -> str:
        return "tsp"

    @property
    def generator_prompt(self) -> str:
        return SYNTHESIS_TEMPLATE

    @property
    def domain_name(self) -> str:
        return TSP_DOMAIN_NAME

    @property
    def domain_guidance(self) -> str:
        return TSP_DOMAIN_GUIDANCE

    @property
    def function_signature(self) -> str:
        return TSP_FUNCTION_SIGNATURE

    @property
    def min_function_args(self) -> int:
        # Agnostic operators take (solution, ctx)
        return 2

    def generate_test_case(self, size: int) -> dict[str, Any]:
        """Generate a random TSP test case."""
        tour = list(range(size))
        random.shuffle(tour)

        dm = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                d = random.uniform(1.0, 100.0)
                dm[i][j] = dm[j][i] = d

        return {"tour": tour, "distance_matrix": dm, "dimension": size}

    def validate_solution(self, result: Any, original: dict[str, Any]) -> bool:
        """Validate that result is a valid TSP tour."""
        expected_len = len(original["tour"])
        if not isinstance(result, list) or len(result) != expected_len:
            return False
        if not all(isinstance(x, int) for x in result):
            return False
        return set(result) == set(range(expected_len))

    def evaluate_fitness(self, solution: Any, instance: dict[str, Any]) -> float:
        """Calculate TSP tour length."""
        tour = solution
        dm = instance["distance_matrix"]
        n = len(tour)
        return sum(dm[tour[i]][tour[(i + 1) % n]] for i in range(n))

    def extract_operator_args(self, input_solution: dict[str, Any]) -> tuple:
        """Extract solution and context for agnostic operator.

        DEPRECATED: Now returns (solution, ctx) for agnostic operators.
        """
        from src.geakg.contexts.tsp import TSPContext

        tour = input_solution["tour"]
        ctx = TSPContext(input_solution["distance_matrix"])
        return (tour, ctx)

    def create_context(self, instance_data: dict[str, Any]) -> "TSPContext":
        """Create TSPContext for agnostic operator execution.

        Args:
            instance_data: Must contain 'distance_matrix'

        Returns:
            TSPContext implementing the 5-method protocol
        """
        from src.geakg.contexts.tsp import TSPContext

        return TSPContext(instance_data["distance_matrix"])


TSP_CONFIG = TSPConfig()
