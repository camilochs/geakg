"""Perturbation operators for TSP.

These operators perturb a solution to escape local optima.
Used in meta-heuristics like ILS, SA, etc.
6 operators total.
"""

import random
from typing import TypeAlias

from src.operators.base import (
    Tour,
    DistanceMatrix,
    calculate_tour_cost,
    get_edges,
)


def double_bridge(
    tour: Tour,
    distance_matrix: DistanceMatrix | None = None,
    **kwargs
) -> Tour:
    """Double bridge move - classic ILS perturbation.

    Removes 4 edges and reconnects in a specific way that
    cannot be reversed by 2-opt or 3-opt.

    The move splits the tour into 4 segments and reconnects them
    in a different order: A-B-C-D becomes A-D-C-B.

    Time complexity: O(n)

    Args:
        tour: Current tour
        distance_matrix: Not used, but accepted for API consistency

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < 8:
        # For small tours, just do a random segment reversal
        if n < 4:
            return tour.copy()
        tour = tour.copy()
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        tour[i:j + 1] = tour[i:j + 1][::-1]
        return tour

    tour = tour.copy()

    # Select 4 random positions that divide tour into 4 segments
    # Ensure segments have minimum size
    min_seg = max(1, n // 8)

    pos1 = random.randint(min_seg, n // 4)
    pos2 = random.randint(pos1 + min_seg, n // 2)
    pos3 = random.randint(pos2 + min_seg, 3 * n // 4)

    # Segments: [0:pos1], [pos1:pos2], [pos2:pos3], [pos3:n]
    seg_a = tour[0:pos1]
    seg_b = tour[pos1:pos2]
    seg_c = tour[pos2:pos3]
    seg_d = tour[pos3:]

    # Reconnect as A-D-C-B
    new_tour = seg_a + seg_d + seg_c + seg_b

    return new_tour


def random_segment_shuffle(
    tour: Tour,
    distance_matrix: DistanceMatrix | None = None,
    n_segments: int = 4,
    **kwargs
) -> Tour:
    """Divide tour into segments and shuffle their order.

    A more aggressive perturbation than double bridge.

    Time complexity: O(n)

    Args:
        tour: Current tour
        distance_matrix: Not used
        n_segments: Number of segments to create

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < n_segments * 2:
        return tour.copy()

    # Create segments
    seg_size = n // n_segments
    segments = []

    for i in range(n_segments - 1):
        segments.append(tour[i * seg_size:(i + 1) * seg_size])
    segments.append(tour[(n_segments - 1) * seg_size:])

    # Shuffle segments
    random.shuffle(segments)

    # Combine
    new_tour = []
    for seg in segments:
        new_tour.extend(seg)

    return new_tour


def guided_mutation(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    edge_frequencies: dict[tuple[int, int], float] | None = None,
    mutation_strength: float = 0.3,
    **kwargs
) -> Tour:
    """Guided mutation based on edge frequency in good solutions.

    Removes edges that appear infrequently in good solutions
    and adds potentially better edges.

    If no edge frequencies provided, removes longest edges.

    Time complexity: O(n²)

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        edge_frequencies: Dict mapping edges to frequency (0-1)
        mutation_strength: Fraction of edges to potentially modify

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    tour = tour.copy()
    n_mutations = max(1, int(n * mutation_strength))

    if edge_frequencies is None:
        # No frequency info: remove longest edges
        # Calculate edge costs
        edge_costs = []
        for i in range(n):
            j = (i + 1) % n
            cost = distance_matrix[tour[i]][tour[j]]
            edge_costs.append((cost, i))

        # Sort by cost (descending) to find longest edges
        edge_costs.sort(reverse=True)

        # Remove some of the longest edges by reversing segments
        for k in range(min(n_mutations, len(edge_costs) // 2)):
            _, pos = edge_costs[k]
            # Random segment reversal involving this position
            seg_len = random.randint(2, max(2, n // 4))
            end_pos = min(pos + seg_len, n - 1)
            tour[pos:end_pos + 1] = tour[pos:end_pos + 1][::-1]
    else:
        # Use edge frequencies
        # Find edges with low frequency
        current_edges = get_edges(tour)
        low_freq_positions = []

        for i in range(n):
            j = (i + 1) % n
            edge = (min(tour[i], tour[j]), max(tour[i], tour[j]))
            freq = edge_frequencies.get(edge, 0.5)
            if freq < 0.5:  # Low frequency edge
                low_freq_positions.append((freq, i))

        low_freq_positions.sort()

        # Perturb around low-frequency edges
        for k in range(min(n_mutations, len(low_freq_positions))):
            _, pos = low_freq_positions[k]
            seg_len = random.randint(2, max(2, n // 4))
            end_pos = min(pos + seg_len, n - 1)
            tour[pos:end_pos + 1] = tour[pos:end_pos + 1][::-1]

    return tour


def ruin_recreate(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    ruin_fraction: float = 0.3,
    recreate_method: str = "greedy",
    **kwargs
) -> Tour:
    """Ruin and Recreate - remove portion of solution and rebuild.

    A powerful diversification operator used in LNS and ALNS.

    Time complexity: O(n²) for greedy reinsertion

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        ruin_fraction: Fraction of cities to remove (0-1)
        recreate_method: "greedy" or "random"

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    n_remove = max(2, int(n * ruin_fraction))
    n_remove = min(n_remove, n - 2)  # Keep at least 2 cities

    # Select cities to remove (random selection)
    remove_indices = set(random.sample(range(n), n_remove))
    removed_cities = [tour[i] for i in remove_indices]

    # Create partial tour
    partial_tour = [tour[i] for i in range(n) if i not in remove_indices]

    if recreate_method == "random":
        # Random reinsertion
        random.shuffle(removed_cities)
        for city in removed_cities:
            pos = random.randint(0, len(partial_tour))
            partial_tour.insert(pos, city)
    else:
        # Greedy reinsertion: insert each city at best position
        for city in removed_cities:
            best_pos = 0
            best_cost = float("inf")

            for pos in range(len(partial_tour) + 1):
                # Calculate insertion cost
                if pos == 0:
                    if len(partial_tour) > 0:
                        cost = (distance_matrix[city][partial_tour[0]] +
                               distance_matrix[partial_tour[-1]][city] -
                               distance_matrix[partial_tour[-1]][partial_tour[0]])
                    else:
                        cost = 0
                elif pos == len(partial_tour):
                    cost = (distance_matrix[partial_tour[-1]][city] +
                           distance_matrix[city][partial_tour[0]] -
                           distance_matrix[partial_tour[-1]][partial_tour[0]])
                else:
                    prev_city = partial_tour[pos - 1]
                    next_city = partial_tour[pos]
                    cost = (distance_matrix[prev_city][city] +
                           distance_matrix[city][next_city] -
                           distance_matrix[prev_city][next_city])

                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

            partial_tour.insert(best_pos, city)

    return partial_tour


def large_neighborhood_search(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    destroy_fraction: float = 0.3,
    **kwargs
) -> Tour:
    """Large Neighborhood Search (LNS) step.

    Combines various destroy and repair operators.
    This implementation uses:
    - Random removal
    - Worst removal (remove worst edges)
    - Greedy repair

    Time complexity: O(n²)

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        destroy_fraction: Fraction to destroy

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    # Choose destroy method randomly
    destroy_method = random.choice(["random", "worst", "related"])

    n_remove = max(2, int(n * destroy_fraction))
    n_remove = min(n_remove, n - 2)

    if destroy_method == "random":
        # Random removal
        remove_indices = set(random.sample(range(n), n_remove))

    elif destroy_method == "worst":
        # Remove cities involved in longest edges
        edge_costs = []
        for i in range(n):
            j = (i + 1) % n
            cost = distance_matrix[tour[i]][tour[j]]
            edge_costs.append((cost, i, tour[i]))

        edge_costs.sort(reverse=True)
        remove_indices = set()

        for cost, idx, city in edge_costs:
            if len(remove_indices) >= n_remove:
                break
            remove_indices.add(idx)

    else:  # related
        # Remove a cluster of related cities
        # Start with random city, add nearest neighbors
        start_idx = random.randint(0, n - 1)
        remove_indices = {start_idx}

        while len(remove_indices) < n_remove:
            # Find city nearest to any removed city
            best_city = None
            best_dist = float("inf")

            for idx in range(n):
                if idx in remove_indices:
                    continue
                for rem_idx in remove_indices:
                    d = distance_matrix[tour[idx]][tour[rem_idx]]
                    if d < best_dist:
                        best_dist = d
                        best_city = idx

            if best_city is None:
                break
            remove_indices.add(best_city)

    removed_cities = [tour[i] for i in remove_indices]
    partial_tour = [tour[i] for i in range(n) if i not in remove_indices]

    # Greedy repair
    for city in removed_cities:
        best_pos = 0
        best_cost = float("inf")

        for pos in range(len(partial_tour) + 1):
            if len(partial_tour) == 0:
                cost = 0
            elif pos == 0:
                cost = (distance_matrix[city][partial_tour[0]] +
                       distance_matrix[partial_tour[-1]][city] -
                       distance_matrix[partial_tour[-1]][partial_tour[0]])
            elif pos == len(partial_tour):
                cost = (distance_matrix[partial_tour[-1]][city] +
                       distance_matrix[city][partial_tour[0]] -
                       distance_matrix[partial_tour[-1]][partial_tour[0]])
            else:
                prev_city = partial_tour[pos - 1]
                next_city = partial_tour[pos]
                cost = (distance_matrix[prev_city][city] +
                       distance_matrix[city][next_city] -
                       distance_matrix[prev_city][next_city])

            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        partial_tour.insert(best_pos, city)

    return partial_tour


def adaptive_mutation(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    initial_rate: float = 0.1,
    success_history: list[bool] | None = None,
    **kwargs
) -> Tour:
    """Adaptive mutation with self-adjusting strength.

    Mutation strength adapts based on recent success/failure.

    Time complexity: O(n)

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        initial_rate: Initial mutation rate
        success_history: List of recent success/failure (True/False)

    Returns:
        Perturbed tour
    """
    n = len(tour)
    if n < 4:
        return tour.copy()

    # Calculate adaptive rate
    if success_history and len(success_history) > 0:
        recent = success_history[-10:]  # Last 10 attempts
        success_rate = sum(recent) / len(recent)

        # If successful: decrease mutation (exploit)
        # If unsuccessful: increase mutation (explore)
        if success_rate > 0.5:
            rate = initial_rate * 0.5  # Less mutation
        elif success_rate < 0.2:
            rate = min(0.5, initial_rate * 2)  # More mutation
        else:
            rate = initial_rate
    else:
        rate = initial_rate

    tour = tour.copy()
    n_moves = max(1, int(n * rate))

    # Apply random moves
    for _ in range(n_moves):
        move_type = random.choice(["swap", "reverse", "insert"])

        if move_type == "swap":
            # Swap two random cities
            i, j = random.sample(range(n), 2)
            tour[i], tour[j] = tour[j], tour[i]

        elif move_type == "reverse":
            # Reverse a random segment
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            tour[i:j + 1] = tour[i:j + 1][::-1]

        else:  # insert
            # Remove and reinsert a city
            i = random.randint(0, n - 1)
            city = tour.pop(i)
            j = random.randint(0, len(tour))
            tour.insert(j, city)

    return tour
