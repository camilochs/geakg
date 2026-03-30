"""VRP Adapter: Transfer TSP operators to VRP domain.

Implementa adaptación TSP→VRP usando representación "giant tour":
- VRP routes: [[1,3], [2,4,5]] (customer IDs 1-indexed)
- TSP tour:   [0,2,1,3,4]       (0-indexed permutation)

El adaptador:
1. Convierte VRP routes a giant tour (flatten + map indices)
2. Crea matriz de distancias para el subgrafo de clientes
3. Aplica operadores TSP al giant tour
4. Reconvierte a routes respetando capacidad

Ejemplo:
    from src.domains.vrp import VRPInstance
    from src.geakg.transfer import VRPAdapter

    adapter = VRPAdapter(vrp_instance)

    # Apply TSP operator to VRP solution
    improved_routes = adapter.apply_operator(
        current_routes,
        two_opt_operator,
    )
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


@dataclass
class VRPRoutes:
    """VRP solution representation."""

    routes: list[list[int]]  # List of routes, each is list of customer IDs (1-indexed)
    cost: float = 0.0


class VRPAdapter(DomainAdapter[VRPRoutes, Any]):
    """Adapter for transferring TSP operators to VRP.

    Representación:
    - VRP: routes = [[c1, c2], [c3, c4, c5]] donde c_i son customer IDs (1 to n)
    - TSP: permutation = [i0, i1, i2, ...] donde i_j son índices (0 to n-1)

    Mapeo:
    - Customer IDs (1-n) ↔ TSP indices (0 to n-1)
    - customers = [1, 2, 3, ...n]
    - idx_to_customer[i] = customers[i]
    - customer_to_idx[c] = i where customers[i] = c
    """

    def __init__(
        self,
        vrp_instance: Any,
        config: AdapterConfig | None = None,
    ):
        """Initialize VRP adapter.

        Args:
            vrp_instance: VRPInstance with distance_matrix, demands, capacity
            config: Optional configuration
        """
        super().__init__(vrp_instance, config)
        self._setup_mappings()

    def _setup_mappings(self) -> None:
        """Setup customer ↔ index mappings."""
        instance = self.target_instance

        # Customers are 1 to n_customers
        self.customers = list(range(1, instance.n_customers + 1))

        # Bidirectional mapping
        self.idx_to_customer = {i: c for i, c in enumerate(self.customers)}
        self.customer_to_idx = {c: i for i, c in enumerate(self.customers)}

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "vrp"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from VRP instance.

        Builds distance matrix for customer-only subgraph.
        """
        instance = self.target_instance
        n = len(self.customers)

        # Create customer-only distance matrix
        # TSP indices: 0 to n-1
        # VRP customers: 1 to n
        tsp_dm = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                c_i = self.idx_to_customer[i]
                c_j = self.idx_to_customer[j]
                # VRP distance matrix includes depot at index 0
                tsp_dm[i][j] = instance.distance_matrix[c_i][c_j]

        # Include extra info for repair/validation
        extra = {
            "demands": instance.demands,  # Original demands array
            "capacity": instance.capacity,
            "depot_distances": [instance.distance_matrix[0][c] for c in self.customers],
            "n_customers": instance.n_customers,
        }

        return SourceContext(
            distance_matrix=tsp_dm,
            index_to_element=self.idx_to_customer.copy(),
            element_to_index=self.customer_to_idx.copy(),
            extra=extra,
        )

    def to_source_repr(
        self,
        vrp_routes: VRPRoutes | list[list[int]],
    ) -> tuple[list[int], SourceContext]:
        """Convert VRP routes to TSP giant tour.

        VRP routes: [[1, 3], [2, 4, 5]]
        TSP tour:   [0, 2, 1, 3, 4]  (mapped indices)

        Args:
            vrp_routes: VRP solution (routes, VRPRoutes, or VRPSolution)

        Returns:
            (tsp_tour, context)
        """
        # Handle VRPRoutes, VRPSolution (from src.domains.vrp), and raw list
        if isinstance(vrp_routes, VRPRoutes):
            routes = vrp_routes.routes
        elif hasattr(vrp_routes, 'routes'):
            # Handle VRPSolution or any object with routes attribute
            routes = vrp_routes.routes
        else:
            routes = vrp_routes

        # Flatten routes to giant tour
        giant_tour = []
        for route in routes:
            for customer in route:
                giant_tour.append(customer)

        # Convert customer IDs to TSP indices
        tsp_tour = [self.customer_to_idx[c] for c in giant_tour]

        context = self.get_context()
        return tsp_tour, context

    def from_source_repr(
        self,
        tsp_tour: list[int],
        context: SourceContext | None = None,
    ) -> VRPRoutes:
        """Convert TSP giant tour back to VRP routes.

        Splits the giant tour into capacity-feasible routes.

        Args:
            tsp_tour: TSP permutation (indices 0 to n-1)
            context: Source context (uses cached if None)

        Returns:
            VRPRoutes with feasible routes
        """
        ctx = context or self.get_context()

        # Convert TSP indices to customer IDs
        giant_tour = [self.idx_to_customer[i] for i in tsp_tour]

        # Split into routes respecting capacity
        routes = self._split_to_routes(giant_tour, ctx)

        # Calculate cost
        cost = self._calculate_cost(routes)

        return VRPRoutes(routes=routes, cost=cost)

    def _split_to_routes(
        self,
        giant_tour: list[int],
        context: SourceContext,
    ) -> list[list[int]]:
        """Split giant tour into capacity-feasible routes.

        Uses greedy split: add customers to current route until
        capacity exceeded, then start new route.

        Args:
            giant_tour: List of customer IDs
            context: Source context with demands and capacity

        Returns:
            List of routes (each route is list of customer IDs)
        """
        demands = context.extra["demands"]
        capacity = context.extra["capacity"]

        routes = []
        current_route = []
        current_load = 0

        for customer in giant_tour:
            demand = demands[customer]

            if current_load + demand <= capacity:
                current_route.append(customer)
                current_load += demand
            else:
                # Start new route
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        # Don't forget last route
        if current_route:
            routes.append(current_route)

        return routes

    def _calculate_cost(self, routes: list[list[int]]) -> float:
        """Calculate total distance of VRP solution.

        Args:
            routes: List of routes

        Returns:
            Total distance including depot visits
        """
        instance = self.target_instance
        dm = instance.distance_matrix
        total = 0.0

        for route in routes:
            if not route:
                continue

            # Depot to first customer
            total += dm[0][route[0]]

            # Between consecutive customers
            for i in range(len(route) - 1):
                total += dm[route[i]][route[i + 1]]

            # Last customer to depot
            total += dm[route[-1]][0]

        return total

    def validate_result(self, result: VRPRoutes) -> bool:
        """Validate VRP solution.

        Checks:
        - All customers visited exactly once
        - No route exceeds capacity
        - Customer IDs are valid

        Args:
            result: VRP solution

        Returns:
            True if valid
        """
        instance = self.target_instance

        visited = set()
        for route in result.routes:
            route_demand = 0

            for customer in route:
                # Check valid customer
                if customer < 1 or customer > instance.n_customers:
                    return False

                # Check not already visited
                if customer in visited:
                    return False

                visited.add(customer)
                route_demand += instance.demands[customer]

            # Check capacity
            if route_demand > instance.capacity:
                return False

        # Check all customers visited
        expected = set(range(1, instance.n_customers + 1))
        return visited == expected

    def _repair_solution(self, solution: VRPRoutes) -> VRPRoutes:
        """Repair invalid VRP solution.

        Strategies:
        1. If missing customers: add to least-loaded route
        2. If duplicate customers: remove from second occurrence
        3. If overcapacity: split route

        Args:
            solution: Potentially invalid solution

        Returns:
            Repaired solution (best effort)
        """
        instance = self.target_instance
        routes = [list(r) for r in solution.routes]  # Copy

        # Find visited customers
        visited = set()
        for route in routes:
            for c in route:
                visited.add(c)

        # Add missing customers
        expected = set(range(1, instance.n_customers + 1))
        missing = expected - visited

        for customer in missing:
            # Add to route with most remaining capacity
            best_route = None
            best_slack = -1

            for route in routes:
                route_demand = sum(instance.demands[c] for c in route)
                slack = instance.capacity - route_demand

                if slack >= instance.demands[customer] and slack > best_slack:
                    best_slack = slack
                    best_route = route

            if best_route is not None:
                best_route.append(customer)
            else:
                # Create new route
                routes.append([customer])

        # Repair capacity violations by splitting
        repaired_routes = []
        for route in routes:
            current = []
            load = 0

            for customer in route:
                demand = instance.demands[customer]
                if load + demand <= instance.capacity:
                    current.append(customer)
                    load += demand
                else:
                    if current:
                        repaired_routes.append(current)
                    current = [customer]
                    load = demand

            if current:
                repaired_routes.append(current)

        cost = self._calculate_cost(repaired_routes)
        return VRPRoutes(routes=repaired_routes, cost=cost)


def create_vrp_adapter(
    vrp_instance: Any,
    split_strategy: str = "greedy",
) -> VRPAdapter:
    """Factory function to create VRP adapter.

    Args:
        vrp_instance: VRPInstance
        split_strategy: How to split giant tour ("greedy", "optimal")

    Returns:
        Configured VRPAdapter
    """
    config = AdapterConfig(
        source_domain="tsp",
        repair_violations=True,
        split_strategy=split_strategy,
    )
    return VRPAdapter(vrp_instance, config)
