"""Local search operators for TSP.

These operators improve an existing tour through local moves.
8 operators total.
"""

import random
from typing import TypeAlias

from src.operators.base import (
    Tour,
    DistanceMatrix,
    calculate_tour_cost,
    reverse_segment,
)


def two_opt(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_iterations: int = 1000,
    first_improvement: bool = False,
    **kwargs
) -> Tour:
    """2-opt local search - remove two edges and reconnect.

    The classic TSP local search operator.
    Reverses a segment of the tour to eliminate crossing edges.

    Time complexity: O(n²) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_iterations: Maximum iterations
        first_improvement: If True, apply first improving move (faster)

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip if would reverse entire tour
                if j == n - 1 and i == 0:
                    continue

                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]

                # Current edges: (a,b) and (c,d)
                # New edges: (a,c) and (b,d)
                current_cost = distance_matrix[a][b] + distance_matrix[c][d]
                new_cost = distance_matrix[a][c] + distance_matrix[b][d]

                if new_cost < current_cost - 1e-10:
                    # Reverse segment between i+1 and j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True

                    if first_improvement:
                        break
            if improved and first_improvement:
                break

    return tour


def three_opt(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_iterations: int = 500,
    **kwargs
) -> Tour:
    """3-opt local search - remove three edges and reconnect.

    More powerful than 2-opt but slower.
    Considers all ways to reconnect after removing 3 edges.

    Time complexity: O(n³) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 6:
        return two_opt(tour, distance_matrix, max_iterations)

    tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    # Get the 6 cities at cut points
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[(k + 1) % n]

                    # Current cost
                    d0 = (distance_matrix[a][b] +
                          distance_matrix[c][d] +
                          distance_matrix[e][f])

                    # Try all 3-opt reconnections (8 possibilities, 4 are useful)
                    # Reconnection 1: 2-opt on first segment
                    d1 = (distance_matrix[a][c] +
                          distance_matrix[b][d] +
                          distance_matrix[e][f])

                    # Reconnection 2: 2-opt on second segment
                    d2 = (distance_matrix[a][b] +
                          distance_matrix[c][e] +
                          distance_matrix[d][f])

                    # Reconnection 3: 2-opt on third segment
                    d3 = (distance_matrix[a][e] +
                          distance_matrix[c][d] +
                          distance_matrix[b][f])

                    # Reconnection 4: True 3-opt (all three segments)
                    d4 = (distance_matrix[a][d] +
                          distance_matrix[e][b] +
                          distance_matrix[c][f])

                    # Reconnection 5: Alternative 3-opt
                    d5 = (distance_matrix[a][c] +
                          distance_matrix[b][e] +
                          distance_matrix[d][f])

                    # Reconnection 6: Alternative 3-opt
                    d6 = (distance_matrix[a][d] +
                          distance_matrix[e][c] +
                          distance_matrix[b][f])

                    # Reconnection 7: Alternative 3-opt
                    d7 = (distance_matrix[a][e] +
                          distance_matrix[d][b] +
                          distance_matrix[c][f])

                    # Find best reconnection
                    best_d = min(d0, d1, d2, d3, d4, d5, d6, d7)

                    if best_d < d0 - 1e-10:
                        # Apply the improvement
                        seg1 = tour[i + 1:j + 1]  # b to c
                        seg2 = tour[j + 1:k + 1]  # d to e

                        if best_d == d1:
                            # Reverse first segment
                            tour[i + 1:j + 1] = seg1[::-1]
                        elif best_d == d2:
                            # Reverse second segment
                            tour[j + 1:k + 1] = seg2[::-1]
                        elif best_d == d3:
                            # Swap segments
                            tour[i + 1:k + 1] = seg2 + seg1
                        elif best_d == d4:
                            # Reverse first, swap
                            tour[i + 1:k + 1] = seg2 + seg1[::-1]
                        elif best_d == d5:
                            # Reverse first segment only
                            tour[i + 1:k + 1] = seg1[::-1] + seg2
                        elif best_d == d6:
                            # Reverse second, swap
                            tour[i + 1:k + 1] = seg2[::-1] + seg1
                        elif best_d == d7:
                            # Both reversed, swapped
                            tour[i + 1:k + 1] = seg2[::-1] + seg1[::-1]

                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return tour


def or_opt(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    segment_sizes: list[int] | None = None,
    max_iterations: int = 500,
    **kwargs
) -> Tour:
    """Or-opt - relocate segments of consecutive cities.

    Moves segments of 1, 2, or 3 consecutive cities to better positions.

    Time complexity: O(n²) per segment size

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        segment_sizes: Sizes of segments to try (default [1, 2, 3])
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    if segment_sizes is None:
        segment_sizes = [1, 2, 3]

    tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for seg_size in segment_sizes:
            if seg_size >= n - 1:
                continue

            for i in range(n):
                # Segment from i to i+seg_size-1
                seg_end = (i + seg_size - 1) % n

                # Cities before and after segment
                prev_i = (i - 1) % n
                next_seg = (seg_end + 1) % n

                # Current cost of edges involving segment
                seg_start_city = tour[i]
                seg_end_city = tour[seg_end]
                prev_city = tour[prev_i]
                next_city = tour[next_seg]

                # Cost of removing segment
                remove_cost = (
                    distance_matrix[prev_city][seg_start_city] +
                    distance_matrix[seg_end_city][next_city]
                )
                reconnect_cost = distance_matrix[prev_city][next_city]

                # Try inserting segment at each other position
                for j in range(n):
                    if j == i or j == prev_i or j == seg_end or j == next_seg:
                        continue

                    next_j = (j + 1) % n
                    if next_j == i:
                        continue

                    city_j = tour[j]
                    city_next_j = tour[next_j]

                    # Cost of inserting segment between j and next_j
                    current_edge = distance_matrix[city_j][city_next_j]
                    insert_cost = (
                        distance_matrix[city_j][seg_start_city] +
                        distance_matrix[seg_end_city][city_next_j]
                    )

                    # Total change
                    delta = (reconnect_cost + insert_cost) - (remove_cost + current_edge)

                    if delta < -1e-10:
                        # Apply move: remove segment and insert at new position
                        segment = []
                        for k in range(seg_size):
                            segment.append(tour[(i + k) % n])

                        # Remove segment
                        new_tour = []
                        idx = next_seg
                        while len(new_tour) < n - seg_size:
                            if idx != i and (i > seg_end or idx < i or idx > seg_end):
                                if not (i <= idx <= seg_end if i <= seg_end else (idx >= i or idx <= seg_end)):
                                    new_tour.append(tour[idx])
                            idx = (idx + 1) % n

                        # Actually, simpler approach: rebuild tour
                        new_tour = [tour[k] for k in range(n) if k < i or k > seg_end] if i <= seg_end else \
                                   [tour[k] for k in range(n) if k > seg_end and k < i]

                        # Find new position for j in new_tour
                        try:
                            new_j = new_tour.index(city_j)
                            new_tour = new_tour[:new_j + 1] + segment + new_tour[new_j + 1:]
                            tour = new_tour
                            improved = True
                            break
                        except ValueError:
                            continue

                if improved:
                    break
            if improved:
                break

    return tour


def swap_operator(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_iterations: int = 500,
    **kwargs
) -> Tour:
    """Swap operator - exchange positions of two cities.

    Time complexity: O(n²) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        best_delta = 0
        best_i, best_j = -1, -1

        for i in range(n):
            for j in range(i + 2, n):
                if j == (i - 1) % n or j == (i + 1) % n:
                    continue

                # Calculate delta for swapping cities at i and j
                prev_i, next_i = (i - 1) % n, (i + 1) % n
                prev_j, next_j = (j - 1) % n, (j + 1) % n

                # Handle adjacent case
                if next_i == j:
                    # i and j are adjacent
                    old_cost = (
                        distance_matrix[tour[prev_i]][tour[i]] +
                        distance_matrix[tour[j]][tour[next_j]]
                    )
                    new_cost = (
                        distance_matrix[tour[prev_i]][tour[j]] +
                        distance_matrix[tour[i]][tour[next_j]]
                    )
                elif prev_i == j:
                    # j and i are adjacent (j before i)
                    old_cost = (
                        distance_matrix[tour[prev_j]][tour[j]] +
                        distance_matrix[tour[i]][tour[next_i]]
                    )
                    new_cost = (
                        distance_matrix[tour[prev_j]][tour[i]] +
                        distance_matrix[tour[j]][tour[next_i]]
                    )
                else:
                    # Non-adjacent
                    old_cost = (
                        distance_matrix[tour[prev_i]][tour[i]] +
                        distance_matrix[tour[i]][tour[next_i]] +
                        distance_matrix[tour[prev_j]][tour[j]] +
                        distance_matrix[tour[j]][tour[next_j]]
                    )
                    new_cost = (
                        distance_matrix[tour[prev_i]][tour[j]] +
                        distance_matrix[tour[j]][tour[next_i]] +
                        distance_matrix[tour[prev_j]][tour[i]] +
                        distance_matrix[tour[i]][tour[next_j]]
                    )

                delta = new_cost - old_cost
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_i >= 0:
            tour[best_i], tour[best_j] = tour[best_j], tour[best_i]
            improved = True

    return tour


def insert_operator(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_iterations: int = 500,
    **kwargs
) -> Tour:
    """Insert/Relocate operator - remove a city and insert at best position.

    Time complexity: O(n²) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        best_delta = 0
        best_city_idx = -1
        best_insert_pos = -1

        for i in range(n):
            city = tour[i]
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            # Cost of removing city i
            remove_gain = (
                distance_matrix[tour[prev_i]][city] +
                distance_matrix[city][tour[next_i]] -
                distance_matrix[tour[prev_i]][tour[next_i]]
            )

            # Try each insertion position
            for j in range(n):
                if j == i or j == prev_i or j == next_i:
                    continue

                next_j = (j + 1) % n
                if next_j == i:
                    continue

                # Cost of inserting after position j
                insert_cost = (
                    distance_matrix[tour[j]][city] +
                    distance_matrix[city][tour[next_j]] -
                    distance_matrix[tour[j]][tour[next_j]]
                )

                delta = insert_cost - remove_gain
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_city_idx = i
                    best_insert_pos = j

        if best_city_idx >= 0:
            # Remove city and insert at new position
            city = tour.pop(best_city_idx)
            # Adjust insert position if needed
            if best_insert_pos > best_city_idx:
                best_insert_pos -= 1
            tour.insert(best_insert_pos + 1, city)
            improved = True

    return tour


def invert_operator(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_iterations: int = 500,
    **kwargs
) -> Tour:
    """Invert/Reverse operator - reverse a segment of the tour.

    Similar to 2-opt but tries all segment lengths.

    Time complexity: O(n²) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    # This is essentially 2-opt
    return two_opt(tour, distance_matrix, max_iterations)


def lin_kernighan(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    max_depth: int = 5,
    max_iterations: int = 100,
    backtracking: bool = True,
    **kwargs
) -> Tour:
    """Lin-Kernighan heuristic - variable-depth search.

    A sophisticated local search that performs variable-depth
    edge exchanges using a sequential search strategy.

    Time complexity: O(n² × depth) per iteration

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        max_depth: Maximum depth of search
        max_iterations: Maximum iterations
        backtracking: Enable backtracking

    Returns:
        Improved tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    tour = tour.copy()
    best_tour = tour.copy()
    best_cost = calculate_tour_cost(tour, distance_matrix)

    def get_neighbors(city: int, k: int = 5) -> list[int]:
        """Get k nearest neighbors of a city."""
        dists = [(distance_matrix[city][j], j) for j in range(n) if j != city]
        dists.sort()
        return [j for _, j in dists[:k]]

    def find_position(t: list[int], city: int) -> int:
        """Find position of city in tour."""
        return t.index(city)

    for iteration in range(max_iterations):
        improved = False

        for start_city in range(n):
            # Start LK move from this city
            t1 = start_city
            pos_t1 = find_position(tour, t1)
            t2 = tour[(pos_t1 + 1) % n]

            # Initial broken edge (t1, t2)
            gain = distance_matrix[t1][t2]

            # Try to find improving sequence
            candidates = get_neighbors(t1, min(n - 1, 10))

            for t3 in candidates:
                if t3 == t2:
                    continue

                # Added edge (t1, t3)
                new_gain = gain - distance_matrix[t1][t3]

                if new_gain <= 0:
                    continue

                # Try closing the tour via t3
                pos_t3 = find_position(tour, t3)
                t4 = tour[(pos_t3 + 1) % n]

                if t4 == t1:
                    continue

                # Check if closing gives improvement
                close_gain = new_gain + distance_matrix[t3][t4] - distance_matrix[t4][t1]

                if close_gain > 1e-10:
                    # Apply 2-opt move between t1-t2 and t3-t4
                    pos_t2 = find_position(tour, t2)
                    if pos_t2 < pos_t3:
                        tour[pos_t2:pos_t3 + 1] = tour[pos_t2:pos_t3 + 1][::-1]
                    else:
                        tour[pos_t3:pos_t2 + 1] = tour[pos_t3:pos_t2 + 1][::-1]

                    current_cost = calculate_tour_cost(tour, distance_matrix)
                    if current_cost < best_cost - 1e-10:
                        best_cost = current_cost
                        best_tour = tour.copy()
                        improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return best_tour


def variable_neighborhood_descent(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    neighborhoods: list[str] | None = None,
    max_iterations: int = 100,
    **kwargs
) -> Tour:
    """Variable Neighborhood Descent (VND).

    Systematically explores multiple neighborhood structures.
    When local optimum is found in one neighborhood, switches to next.

    Time complexity: Depends on neighborhoods used

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        neighborhoods: List of neighborhood operators to use
        max_iterations: Maximum iterations

    Returns:
        Improved tour
    """
    if neighborhoods is None:
        neighborhoods = ["swap", "insert", "two_opt"]

    # Map names to functions
    neighborhood_funcs = {
        "swap": lambda t: swap_operator(t, distance_matrix, max_iterations=50),
        "insert": lambda t: insert_operator(t, distance_matrix, max_iterations=50),
        "two_opt": lambda t: two_opt(t, distance_matrix, max_iterations=100),
        "or_opt": lambda t: or_opt(t, distance_matrix, max_iterations=50),
        "three_opt": lambda t: three_opt(t, distance_matrix, max_iterations=20),
    }

    tour = tour.copy()
    current_cost = calculate_tour_cost(tour, distance_matrix)
    iterations = 0

    while iterations < max_iterations:
        improved = False
        iterations += 1

        for neighborhood_name in neighborhoods:
            if neighborhood_name not in neighborhood_funcs:
                continue

            func = neighborhood_funcs[neighborhood_name]
            new_tour = func(tour)
            new_cost = calculate_tour_cost(new_tour, distance_matrix)

            if new_cost < current_cost - 1e-10:
                tour = new_tour
                current_cost = new_cost
                improved = True
                break  # Restart from first neighborhood

        if not improved:
            break  # Local optimum in all neighborhoods

    return tour
