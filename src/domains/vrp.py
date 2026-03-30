"""Vehicle Routing Problem (VRP) domain implementation.

Supports:
- CVRP: Capacitated Vehicle Routing Problem
- Distance minimization with capacity constraints
"""

import math
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from src.domains.base import OptimizationDomain, ProblemFeatures


class VRPInstance(BaseModel):
    """VRP problem instance.

    A VRP instance consists of:
    - A depot (node 0)
    - n customers (nodes 1 to n)
    - Distance matrix between all nodes
    - Demands for each customer
    - Vehicle capacity
    """

    name: str
    n_customers: int = Field(gt=0)
    capacity: int = Field(gt=0)
    # demands[i] = demand of customer i (depot has demand 0)
    demands: list[int]
    # coordinates[i] = (x, y) for node i
    coordinates: list[tuple[float, float]] | None = None
    # distance_matrix[i][j] = distance from node i to node j
    distance_matrix: list[list[float]]
    optimal_cost: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def dimension(self) -> int:
        """Total number of nodes (depot + customers)."""
        return self.n_customers + 1

    @computed_field
    @property
    def total_demand(self) -> int:
        """Total demand of all customers."""
        return sum(self.demands)

    @computed_field
    @property
    def min_vehicles(self) -> int:
        """Minimum number of vehicles needed."""
        return math.ceil(self.total_demand / self.capacity)


class VRPSolution(BaseModel):
    """VRP solution (set of routes).

    Each route is a list of customer indices (not including depot).
    The depot (0) is implicitly at the start and end of each route.
    """

    routes: list[list[int]]  # List of routes, each route is list of customers
    cost: float = Field(ge=0, default=0.0)
    is_valid: bool = True

    @property
    def n_routes(self) -> int:
        """Number of routes (vehicles used)."""
        return len(self.routes)

    @property
    def all_customers(self) -> set[int]:
        """Set of all customers served."""
        return {c for route in self.routes for c in route}


class VRPFeatures(ProblemFeatures):
    """Features extracted from VRP instance."""

    dimension: int
    n_customers: int
    capacity: int
    total_demand: int
    avg_demand: float
    std_demand: float
    demand_capacity_ratio: float
    avg_distance_to_depot: float
    avg_inter_customer_distance: float
    clustering_coefficient: float

    @classmethod
    def from_instance(cls, instance: VRPInstance) -> "VRPFeatures":
        """Extract features from VRP instance."""
        n = instance.n_customers
        demands = instance.demands[1:]  # Exclude depot

        avg_demand = sum(demands) / n if n > 0 else 0
        variance = sum((d - avg_demand) ** 2 for d in demands) / n if n > 0 else 0
        std_demand = variance ** 0.5

        # Distance to depot
        depot_distances = [instance.distance_matrix[0][i] for i in range(1, instance.dimension)]
        avg_depot_dist = sum(depot_distances) / len(depot_distances) if depot_distances else 0

        # Inter-customer distances
        inter_dists = []
        for i in range(1, instance.dimension):
            for j in range(i + 1, instance.dimension):
                inter_dists.append(instance.distance_matrix[i][j])
        avg_inter_dist = sum(inter_dists) / len(inter_dists) if inter_dists else 0

        # Clustering: fraction of customers close to each other
        if inter_dists:
            threshold = sorted(inter_dists)[len(inter_dists) // 4]  # 25th percentile
            close_pairs = sum(1 for d in inter_dists if d < threshold)
            clustering = close_pairs / len(inter_dists)
        else:
            clustering = 0

        return cls(
            dimension=instance.dimension,
            n_customers=n,
            capacity=instance.capacity,
            total_demand=instance.total_demand,
            avg_demand=avg_demand,
            std_demand=std_demand,
            demand_capacity_ratio=instance.total_demand / (instance.capacity * n) if n > 0 else 0,
            avg_distance_to_depot=avg_depot_dist,
            avg_inter_customer_distance=avg_inter_dist,
            clustering_coefficient=clustering,
        )


class VRPDomain(OptimizationDomain[VRPInstance, VRPSolution]):
    """VRP domain implementation."""

    @property
    def name(self) -> str:
        return "vrp"

    def load_instance(self, path: Path) -> VRPInstance:
        """Load VRP instance from CVRPLIB format.

        Args:
            path: Path to instance file

        Returns:
            Loaded VRP instance
        """
        with open(path) as f:
            content = f.read()

        return self._parse_vrplib(content, path.stem)

    def _parse_vrplib(self, content: str, name: str) -> VRPInstance:
        """Parse VRPLIB format content."""
        lines = [l.strip() for l in content.strip().split("\n")]

        dimension = 0
        capacity = 0
        coordinates = []
        demands = []
        optimal = None

        section = None

        for line in lines:
            if not line:
                continue
            if line.startswith("COMMENT"):
                # Extract optimal value from comment like "Optimal value: 784"
                if "Optimal" in line or "optimal" in line:
                    match = re.search(r"Optimal[^:]*:\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
                    if match:
                        optimal = float(match.group(1))
                continue

            if line.startswith("NAME"):
                continue
            elif line.startswith("TYPE"):
                continue
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                continue
            elif line.startswith("NODE_COORD_SECTION"):
                section = "coords"
            elif line.startswith("DEMAND_SECTION"):
                section = "demands"
            elif line.startswith("DEPOT_SECTION"):
                section = "depot"
            elif line == "EOF":
                break
            elif section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append((x, y))
            elif section == "demands":
                parts = line.split()
                if len(parts) >= 2:
                    demands.append(int(parts[1]))
            elif section == "depot":
                continue

        # Build distance matrix from coordinates
        n = len(coordinates)
        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

        return VRPInstance(
            name=name,
            n_customers=dimension - 1,  # Exclude depot
            capacity=capacity,
            demands=demands if demands else [0] * dimension,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            optimal_cost=optimal,
        )

    def evaluate_solution(self, solution: VRPSolution, instance: VRPInstance) -> float:
        """Compute total distance of all routes.

        Args:
            solution: VRP solution (routes)
            instance: VRP instance

        Returns:
            Total distance (lower is better)
        """
        total_distance = 0.0

        for route in solution.routes:
            if not route:
                continue
            # Distance from depot to first customer
            total_distance += instance.distance_matrix[0][route[0]]
            # Distance between consecutive customers
            for i in range(len(route) - 1):
                total_distance += instance.distance_matrix[route[i]][route[i + 1]]
            # Distance from last customer back to depot
            total_distance += instance.distance_matrix[route[-1]][0]

        return total_distance

    def validate_solution(self, solution: VRPSolution, instance: VRPInstance) -> bool:
        """Check if solution is valid.

        Validates:
        - All customers are visited exactly once
        - No route exceeds capacity
        - Customer indices are valid

        Args:
            solution: VRP solution
            instance: VRP instance

        Returns:
            True if valid
        """
        visited = set()

        for route in solution.routes:
            route_demand = 0
            for customer in route:
                # Check valid customer index
                if customer < 1 or customer >= instance.dimension:
                    return False
                # Check not visited before
                if customer in visited:
                    return False
                visited.add(customer)
                route_demand += instance.demands[customer]

            # Check capacity constraint
            if route_demand > instance.capacity:
                return False

        # Check all customers visited
        expected = set(range(1, instance.dimension))
        if visited != expected:
            return False

        return True

    def get_features(self, instance: VRPInstance) -> VRPFeatures:
        """Extract features from VRP instance.

        Args:
            instance: VRP instance

        Returns:
            Extracted features
        """
        return VRPFeatures.from_instance(instance)

    def random_solution(self, instance: VRPInstance) -> VRPSolution:
        """Generate a random valid solution.

        Uses random assignment to routes respecting capacity.

        Args:
            instance: VRP instance

        Returns:
            Random valid solution
        """
        customers = list(range(1, instance.dimension))
        random.shuffle(customers)

        routes = []
        current_route = []
        current_load = 0

        for customer in customers:
            demand = instance.demands[customer]
            if current_load + demand <= instance.capacity:
                current_route.append(customer)
                current_load += demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        if current_route:
            routes.append(current_route)

        solution = VRPSolution(routes=routes)
        solution.cost = self.evaluate_solution(solution, instance)
        return solution

    def savings_solution(self, instance: VRPInstance) -> VRPSolution:
        """Generate solution using Clarke-Wright Savings algorithm.

        Args:
            instance: VRP instance

        Returns:
            Solution from savings algorithm
        """
        n = instance.dimension

        # Calculate savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = (instance.distance_matrix[0][i] +
                     instance.distance_matrix[0][j] -
                     instance.distance_matrix[i][j])
                savings.append((s, i, j))

        # Sort by savings (descending)
        savings.sort(reverse=True)

        # Initialize: each customer in own route
        routes = {i: [i] for i in range(1, n)}
        route_demand = {i: instance.demands[i] for i in range(1, n)}
        customer_route = {i: i for i in range(1, n)}

        # Merge routes based on savings
        for s, i, j in savings:
            route_i = customer_route[i]
            route_j = customer_route[j]

            if route_i == route_j:
                continue

            # Check if merge is feasible
            combined_demand = route_demand[route_i] + route_demand[route_j]
            if combined_demand > instance.capacity:
                continue

            # Check if i and j are at route ends
            ri = routes[route_i]
            rj = routes[route_j]

            if ri[-1] == i and rj[0] == j:
                # Merge: route_i + route_j
                new_route = ri + rj
            elif ri[0] == i and rj[-1] == j:
                # Merge: route_j + route_i
                new_route = rj + ri
            elif ri[-1] == i and rj[-1] == j:
                # Merge: route_i + reversed(route_j)
                new_route = ri + list(reversed(rj))
            elif ri[0] == i and rj[0] == j:
                # Merge: reversed(route_i) + route_j
                new_route = list(reversed(ri)) + rj
            else:
                continue

            # Perform merge
            routes[route_i] = new_route
            route_demand[route_i] = combined_demand
            del routes[route_j]
            del route_demand[route_j]

            for c in new_route:
                customer_route[c] = route_i

        result_routes = list(routes.values())
        solution = VRPSolution(routes=result_routes)
        solution.cost = self.evaluate_solution(solution, instance)
        return solution


def create_sample_vrp_instance(
    n_customers: int = 10,
    capacity: int = 100,
    seed: int = 42,
) -> VRPInstance:
    """Create a sample VRP instance for testing.

    Args:
        n_customers: Number of customers
        capacity: Vehicle capacity
        seed: Random seed

    Returns:
        Sample VRP instance
    """
    random.seed(seed)

    # Generate coordinates (depot at center)
    coordinates = [(50.0, 50.0)]  # Depot
    for _ in range(n_customers):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        coordinates.append((x, y))

    # Generate demands (depot has 0)
    demands = [0]  # Depot
    for _ in range(n_customers):
        demands.append(random.randint(5, 30))

    # Build distance matrix
    n = len(coordinates)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)

    return VRPInstance(
        name=f"sample_vrp_{n_customers}",
        n_customers=n_customers,
        capacity=capacity,
        demands=demands,
        coordinates=coordinates,
        distance_matrix=distance_matrix,
    )
