"""VRP domain context adapter.

Implements DomainContext protocol for Vehicle Routing Problem.

Lampson principle applied: "Keep secrets"
- distances, demands, capacity are hidden from operators
- Operators only see cost(), delta(), neighbors(), evaluate(), valid()

Note: VRP solutions use flat list representation for compatibility with
domain-agnostic operators. Routes are separated by depot (index 0).
"""

from __future__ import annotations


class VRPContext:
    """Adapts VRP instance to DomainContext interface.

    The distances, demands, and capacity are SECRETS - operators never access them.
    All VRP-specific operations go through the 5 protocol methods.

    Solution representation:
        Flat list with depot (0) as separator: [1, 3, 5, 0, 2, 4, 0, 6, 7]
        Represents routes: [1,3,5], [2,4], [6,7]

    Example:
        >>> dist = [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]]
        >>> demands = [0, 3, 4, 5]  # depot has 0 demand
        >>> ctx = VRPContext(dist, demands, capacity=10)
        >>> solution = [1, 2, 0, 3]  # Two routes: [1,2] and [3]
        >>> ctx.evaluate(solution)
    """

    def __init__(
        self,
        distances: list[list[float]],
        demands: list[int],
        capacity: int,
    ):
        """Initialize VRP context.

        Args:
            distances: NxN distance matrix (index 0 is depot)
            demands: Demand for each customer (index 0 = depot, demand 0)
            capacity: Vehicle capacity
        """
        self._dist = distances  # SECRET
        self._demands = demands  # SECRET
        self._cap = capacity  # SECRET
        self._n = len(distances)  # Total nodes including depot
        self._n_customers = self._n - 1  # Customers only

    def cost(self, solution: list[int], i: int) -> float:
        """Cost contribution of element at position i.

        Cost = distance from previous + distance to next.
        Depots (0) at boundaries connect to actual depot.

        Args:
            solution: Flat list with depot separators
            i: Position in solution

        Returns:
            Distance contribution of element at position i
        """
        if not solution or i >= len(solution):
            return 0.0

        n = len(solution)
        node = solution[i]

        # Get previous and next nodes
        if i == 0:
            prev_node = 0  # Start from depot
        else:
            prev_node = solution[i - 1]
            if prev_node == 0:
                prev_node = 0  # Previous is depot

        if i == n - 1:
            next_node = 0  # Return to depot
        else:
            next_node = solution[i + 1]
            if next_node == 0:
                next_node = 0  # Next is depot

        # If this is a depot separator, cost is 0 (handled by neighbors)
        if node == 0:
            return 0.0

        return self._dist[prev_node][node] + self._dist[node][next_node]

    def delta(self, solution: list[int], move: str, i: int, j: int) -> float:
        """Calculate delta cost for a move without executing it.

        Args:
            solution: Current solution
            move: Move type ("swap", "relocate", "2opt")
            i: First position
            j: Second position

        Returns:
            Change in total cost (negative = improvement)
        """
        if move == "swap":
            return self._swap_delta(solution, i, j)
        elif move == "relocate":
            return self._relocate_delta(solution, i, j)
        elif move == "2opt":
            return self._2opt_delta(solution, i, j)
        return 0.0

    def _swap_delta(self, solution: list[int], i: int, j: int) -> float:
        """Delta for swapping positions i and j."""
        if i == j:
            return 0.0

        # Don't swap depot separators
        if solution[i] == 0 or solution[j] == 0:
            return float("inf")  # Invalid move

        # Calculate current cost
        old_cost = self.cost(solution, i) + self.cost(solution, j)

        # Temporarily swap
        solution[i], solution[j] = solution[j], solution[i]
        new_cost = self.cost(solution, i) + self.cost(solution, j)
        # Restore
        solution[i], solution[j] = solution[j], solution[i]

        # Check capacity after swap
        if not self._check_route_capacity_at(solution, i) or \
           not self._check_route_capacity_at(solution, j):
            return float("inf")

        return new_cost - old_cost

    def _relocate_delta(self, solution: list[int], i: int, j: int) -> float:
        """Delta for relocating element from i to j."""
        if i == j:
            return 0.0

        if solution[i] == 0:
            return float("inf")  # Can't relocate depot

        # Simplified: use full evaluation
        old_cost = self.evaluate(solution)

        # Temporarily relocate
        elem = solution[i]
        temp = solution[:i] + solution[i + 1:]
        if j > i:
            j -= 1
        temp = temp[:j] + [elem] + temp[j:]

        if not self.valid(temp):
            return float("inf")

        new_cost = self.evaluate(temp)
        return new_cost - old_cost

    def _2opt_delta(self, solution: list[int], i: int, j: int) -> float:
        """Delta for 2-opt within a route segment."""
        # Simplified implementation
        if i == j:
            return 0.0

        old_cost = self.evaluate(solution)

        # Create reversed segment
        if i > j:
            i, j = j, i
        temp = solution[:i] + solution[i:j + 1][::-1] + solution[j + 1:]

        if not self.valid(temp):
            return float("inf")

        new_cost = self.evaluate(temp)
        return new_cost - old_cost

    def _check_route_capacity_at(self, solution: list[int], pos: int) -> bool:
        """Check if route containing position pos respects capacity."""
        # Find route boundaries
        start = pos
        while start > 0 and solution[start - 1] != 0:
            start -= 1

        end = pos
        while end < len(solution) - 1 and solution[end + 1] != 0:
            end += 1

        # Sum demands
        total_demand = sum(self._demands[solution[k]]
                          for k in range(start, end + 1)
                          if solution[k] != 0)

        return total_demand <= self._cap

    def neighbors(self, solution: list[int], i: int, k: int) -> list[int]:
        """K nearest positions by distance to node at position i.

        Args:
            solution: Current solution
            i: Position in solution
            k: Number of neighbors to return

        Returns:
            List of k position indices with nearest customers
        """
        if not solution or i >= len(solution):
            return []

        node = solution[i]
        if node == 0:
            return []  # Depot has no meaningful neighbors

        n = len(solution)
        distances = []

        for j in range(n):
            if j != i and solution[j] != 0:
                other_node = solution[j]
                dist = self._dist[node][other_node]
                distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        return [pos for pos, _ in distances[:k]]

    def evaluate(self, solution: list[int]) -> float:
        """Calculate total route distance.

        Args:
            solution: Flat list with depot separators

        Returns:
            Total distance of all routes
        """
        if not solution:
            return 0.0

        total = 0.0
        prev = 0  # Start at depot

        for node in solution:
            total += self._dist[prev][node]
            prev = node if node != 0 else 0

        # Return to depot from last customer
        if prev != 0:
            total += self._dist[prev][0]

        return total

    def valid(self, solution: list[int]) -> bool:
        """Check if solution is valid VRP solution.

        Valid solution:
        - All customers visited exactly once
        - Each route respects capacity

        Args:
            solution: Solution to validate

        Returns:
            True if solution is valid
        """
        if not solution:
            return self._n_customers == 0

        # Check all customers visited exactly once
        customers = [x for x in solution if x != 0]
        if len(customers) != self._n_customers:
            return False
        if set(customers) != set(range(1, self._n + 1)) - {0}:
            # Might have different indexing, check we have right count
            if len(set(customers)) != self._n_customers:
                return False

        # Check capacity for each route
        route_demand = 0
        for node in solution:
            if node == 0:
                # End of route, check capacity
                if route_demand > self._cap:
                    return False
                route_demand = 0
            else:
                route_demand += self._demands[node]

        # Check last route
        if route_demand > self._cap:
            return False

        return True

    @property
    def size(self) -> int:
        """Number of customers (excluding depot)."""
        return self._n_customers

    @property
    def capacity(self) -> int:
        """Vehicle capacity (exposed for operators that need it)."""
        return self._cap
