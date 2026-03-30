"""TSP domain context adapter.

Implements PermutationContext for Traveling Salesman Problem.

Lampson principle applied: "Keep secrets"
- distance_matrix is hidden from operators
- Operators only see cost(), delta(), neighbors(), evaluate(), valid()
"""

from __future__ import annotations

from src.geakg.contexts.families.permutation import PermutationContext


class TSPContext(PermutationContext):
    """Adapts TSP instance to PermutationContext interface.

    The distance_matrix is a SECRET - operators never access it directly.
    All distance-based operations go through the protocol methods.

    Example:
        >>> dm = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
        >>> ctx = TSPContext(dm)
        >>> tour = [0, 1, 2]
        >>> ctx.evaluate(tour)  # Returns tour length
        45.0
        >>> ctx.cost(tour, 1)   # Cost contribution of city 1
        30.0
    """

    def __init__(self, distance_matrix: list[list[float]]):
        """Initialize TSP context.

        Args:
            distance_matrix: NxN symmetric distance matrix
        """
        self._dm = distance_matrix  # SECRET - never exposed to operators
        self._n = len(distance_matrix)

    @property
    def domain(self) -> str:
        """Domain identifier."""
        return "tsp"

    @property
    def dimension(self) -> int:
        """Number of cities."""
        return self._n

    # Backward compatibility alias
    @property
    def size(self) -> int:
        """Number of cities in the instance (alias for dimension)."""
        return self._n

    def cost(self, tour: list[int], i: int) -> float:
        """Cost contribution of city at position i.

        Cost = distance from previous city + distance to next city.

        Args:
            tour: Current tour as list of city indices
            i: Position in tour (0 to n-1)

        Returns:
            Sum of incoming and outgoing edge costs
        """
        n = len(tour)
        if n == 0:
            return 0.0

        prev_i = (i - 1) % n
        next_i = (i + 1) % n

        city = tour[i]
        prev_city = tour[prev_i]
        next_city = tour[next_i]

        return self._dm[prev_city][city] + self._dm[city][next_city]

    def delta(self, tour: list[int], move: str, i: int, j: int) -> float:
        """Calculate delta cost for a move without executing it.

        Efficient O(1) calculation for common moves.

        Args:
            tour: Current tour
            move: Move type ("swap", "reverse"/"2opt", "insert")
            i: First position
            j: Second position

        Returns:
            Change in total cost (negative = improvement)
        """
        if move == "swap":
            return self._swap_delta(tour, i, j)
        elif move in ("2opt", "reverse"):
            return self._2opt_delta(tour, i, j)
        elif move == "insert":
            return self._insert_delta(tour, i, j)
        return 0.0

    def delta_swap(self, solution: list[int], i: int, j: int) -> float:
        """O(1) delta for swap move."""
        return self._swap_delta(solution, i, j)

    def delta_reverse(self, solution: list[int], i: int, j: int) -> float:
        """O(1) delta for 2-opt (reverse) move."""
        return self._2opt_delta(solution, i, j)

    def _swap_delta(self, tour: list[int], i: int, j: int) -> float:
        """Delta for swapping positions i and j."""
        if i == j:
            return 0.0

        n = len(tour)
        if n < 3:
            return 0.0

        # Ensure i < j
        if i > j:
            i, j = j, i

        # Handle adjacent swap specially
        if j == i + 1 or (i == 0 and j == n - 1):
            return self._adjacent_swap_delta(tour, i, j)

        # Non-adjacent swap: calculate old and new costs
        old_cost = self.cost(tour, i) + self.cost(tour, j)

        # Temporarily swap
        tour[i], tour[j] = tour[j], tour[i]
        new_cost = self.cost(tour, i) + self.cost(tour, j)
        # Restore
        tour[i], tour[j] = tour[j], tour[i]

        return new_cost - old_cost

    def _adjacent_swap_delta(self, tour: list[int], i: int, j: int) -> float:
        """Delta for swapping adjacent positions."""
        n = len(tour)

        # Get cities
        city_i = tour[i]
        city_j = tour[j]

        # Get neighbors
        prev_i = tour[(i - 1) % n]
        next_j = tour[(j + 1) % n]

        # Old cost: prev_i -> city_i -> city_j -> next_j
        old_cost = (
            self._dm[prev_i][city_i]
            + self._dm[city_i][city_j]
            + self._dm[city_j][next_j]
        )

        # New cost: prev_i -> city_j -> city_i -> next_j
        new_cost = (
            self._dm[prev_i][city_j]
            + self._dm[city_j][city_i]
            + self._dm[city_i][next_j]
        )

        return new_cost - old_cost

    def _2opt_delta(self, tour: list[int], i: int, j: int) -> float:
        """Delta for 2-opt move (reverse segment from i to j)."""
        if i == j:
            return 0.0

        n = len(tour)
        if i > j:
            i, j = j, i

        # Cities involved
        city_i = tour[i]
        city_j = tour[j]
        prev_i = tour[(i - 1) % n]
        next_j = tour[(j + 1) % n]

        # Old edges: (prev_i, city_i) and (city_j, next_j)
        old_cost = self._dm[prev_i][city_i] + self._dm[city_j][next_j]

        # New edges: (prev_i, city_j) and (city_i, next_j)
        new_cost = self._dm[prev_i][city_j] + self._dm[city_i][next_j]

        return new_cost - old_cost

    def _insert_delta(self, tour: list[int], i: int, j: int) -> float:
        """Delta for inserting element at i to position j."""
        if i == j or i == j - 1:
            return 0.0

        n = len(tour)

        # Current edges involving position i
        prev_i = tour[(i - 1) % n]
        city_i = tour[i]
        next_i = tour[(i + 1) % n]

        # Cost of removing city_i
        remove_cost = (
            self._dm[prev_i][city_i]
            + self._dm[city_i][next_i]
            - self._dm[prev_i][next_i]
        )

        # Cost of inserting at position j
        if j > i:
            # Adjust j since we're removing i first
            actual_j = j - 1
        else:
            actual_j = j

        # Get neighbors at insertion point
        if actual_j == 0:
            prev_j = tour[n - 1] if i != n - 1 else tour[n - 2]
            next_j = tour[0] if i != 0 else tour[1]
        else:
            prev_j = tour[actual_j - 1] if actual_j - 1 != i else tour[actual_j - 2]
            next_j = tour[actual_j] if actual_j != i else tour[actual_j + 1]

        insert_cost = (
            self._dm[prev_j][city_i]
            + self._dm[city_i][next_j]
            - self._dm[prev_j][next_j]
        )

        return insert_cost - remove_cost

    def neighbors(self, tour: list[int], i: int, k: int) -> list[int]:
        """K nearest positions by distance to city at position i.

        Args:
            tour: Current tour
            i: Position in tour
            k: Number of neighbors to return

        Returns:
            List of k position indices with nearest cities
        """
        n = len(tour)
        if k >= n - 1:
            return [j for j in range(n) if j != i]

        city = tour[i]

        # Calculate distances to all other positions
        distances = []
        for j in range(n):
            if j != i:
                other_city = tour[j]
                dist = self._dm[city][other_city]
                distances.append((j, dist))

        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return [pos for pos, _ in distances[:k]]

    def evaluate(self, tour: list[int]) -> float:
        """Calculate total tour length.

        Args:
            tour: Tour as list of city indices

        Returns:
            Total tour length (sum of all edge costs)
        """
        if len(tour) < 2:
            return 0.0

        total = 0.0
        n = len(tour)
        for i in range(n):
            total += self._dm[tour[i]][tour[(i + 1) % n]]

        return total

    @property
    def instance(self) -> dict:
        """Instance data for construction operators that need direct access."""
        return {"distance_matrix": self._dm, "dimension": self._n}

    @property
    def instance_data(self) -> dict:
        """Instance data (alias for instance property)."""
        return self.instance
