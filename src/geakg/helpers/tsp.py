"""TSP helper functions for synthesized operators.

These helpers are injected into the operator's namespace at runtime.
They provide a clean interface that:
1. Makes operator code more readable
2. Enables transfer learning (same function names, different implementations per domain)

The helpers use (tour, dm, ...) signature to match what the LLM sees,
but internally they map to the same concepts as ctx methods.
"""


def tour_cost(tour: list[int], dm: list[list[float]]) -> float:
    """Calculate total tour length.

    Args:
        tour: List of city indices representing the tour
        dm: Distance matrix where dm[i][j] = distance from city i to j

    Returns:
        Total tour cost (sum of all edge distances)
    """
    n = len(tour)
    return sum(dm[tour[i]][tour[(i + 1) % n]] for i in range(n))


def position_cost(tour: list[int], dm: list[list[float]], i: int) -> float:
    """Cost contribution of city at position i.

    This measures how "expensive" the city at position i is,
    based on its edges to neighboring cities in the tour.

    Args:
        tour: Current tour
        dm: Distance matrix
        i: Position index (0 to n-1)

    Returns:
        Sum of distances to previous and next city in tour
    """
    n = len(tour)
    prev_pos = (i - 1) % n
    next_pos = (i + 1) % n
    return dm[tour[prev_pos]][tour[i]] + dm[tour[i]][tour[next_pos]]


def delta_swap(tour: list[int], dm: list[list[float]], i: int, j: int) -> float:
    """Calculate cost change if positions i and j are swapped.

    Does NOT modify the tour - only calculates the delta.

    Args:
        tour: Current tour
        dm: Distance matrix
        i, j: Positions to swap

    Returns:
        Cost change (negative = improvement)
    """
    if i == j:
        return 0.0

    n = len(tour)

    # Handle adjacent positions specially
    if abs(i - j) == 1 or abs(i - j) == n - 1:
        # Adjacent swap - simpler calculation
        old_cost = position_cost(tour, dm, i) + position_cost(tour, dm, j)
        # Temporarily swap
        tour[i], tour[j] = tour[j], tour[i]
        new_cost = position_cost(tour, dm, i) + position_cost(tour, dm, j)
        # Restore
        tour[i], tour[j] = tour[j], tour[i]
        # Avoid double counting shared edge
        return (new_cost - old_cost) / 2

    # Non-adjacent swap
    old_cost = position_cost(tour, dm, i) + position_cost(tour, dm, j)
    # Temporarily swap
    tour[i], tour[j] = tour[j], tour[i]
    new_cost = position_cost(tour, dm, i) + position_cost(tour, dm, j)
    # Restore
    tour[i], tour[j] = tour[j], tour[i]

    return new_cost - old_cost


def delta_2opt(tour: list[int], dm: list[list[float]], i: int, j: int) -> float:
    """Calculate cost change for 2-opt move (reverse segment from i+1 to j).

    2-opt removes edges (i, i+1) and (j, j+1), adds edges (i, j) and (i+1, j+1).

    Args:
        tour: Current tour
        dm: Distance matrix
        i, j: Positions defining the segment to reverse (reverses i+1 to j)

    Returns:
        Cost change (negative = improvement)
    """
    if i >= j:
        return 0.0

    n = len(tour)
    a, b = tour[i], tour[(i + 1) % n]
    c, d = tour[j], tour[(j + 1) % n]

    # Remove edges a-b and c-d, add edges a-c and b-d
    return (dm[a][c] + dm[b][d]) - (dm[a][b] + dm[c][d])


def nearest_positions(tour: list[int], dm: list[list[float]], i: int, k: int) -> list[int]:
    """Find k positions with cities nearest to the city at position i.

    Useful for focused local search - check promising swaps first.

    Args:
        tour: Current tour
        dm: Distance matrix
        i: Position index
        k: Number of nearest positions to return

    Returns:
        List of position indices sorted by distance to city at i
    """
    n = len(tour)
    city_i = tour[i]

    # Calculate distances from city_i to all other cities in tour
    distances = []
    for j in range(n):
        if j != i:
            city_j = tour[j]
            distances.append((j, dm[city_i][city_j]))

    # Sort by distance and return top k positions
    distances.sort(key=lambda x: x[1])
    return [pos for pos, _ in distances[:k]]


# Names to inject into operator namespace
TSP_HELPERS = {
    "tour_cost": tour_cost,
    "position_cost": position_cost,
    "delta_swap": delta_swap,
    "delta_2opt": delta_2opt,
    "nearest_positions": nearest_positions,
}


def get_tsp_helper_namespace() -> dict:
    """Get namespace dict with all TSP helpers.

    Returns:
        Dict mapping helper names to functions
    """
    return TSP_HELPERS.copy()
