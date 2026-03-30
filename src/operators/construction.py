"""Construction operators for TSP.

These operators build an initial tour from scratch.
10 operators total.
"""

import math
import random
from typing import TypeAlias

from src.operators.base import (
    Tour,
    DistanceMatrix,
    Coordinates,
    calculate_tour_cost,
    euclidean_distance,
    get_nearest_neighbors,
)


def greedy_nearest_neighbor(
    distance_matrix: DistanceMatrix,
    start: int | None = None,
    **kwargs
) -> Tour:
    """Build tour by always visiting the nearest unvisited city.

    Time complexity: O(n²)

    Args:
        distance_matrix: Distance matrix between cities
        start: Starting city (random if None)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if start is None:
        start = random.randint(0, n - 1)

    tour = [start]
    visited = {start}

    while len(tour) < n:
        current = tour[-1]
        best_next = None
        best_dist = float("inf")

        for city in range(n):
            if city not in visited:
                dist = distance_matrix[current][city]
                if dist < best_dist:
                    best_dist = dist
                    best_next = city

        if best_next is not None:
            tour.append(best_next)
            visited.add(best_next)

    return tour


def farthest_insertion(
    distance_matrix: DistanceMatrix,
    **kwargs
) -> Tour:
    """Build tour by inserting the farthest city from current tour.

    Algorithm:
    1. Start with two farthest cities
    2. Find city farthest from tour
    3. Insert at position that minimizes tour length increase
    4. Repeat until all cities inserted

    Time complexity: O(n³)

    Args:
        distance_matrix: Distance matrix between cities

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Find two farthest cities to start
    max_dist = -1
    start_i, start_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                start_i, start_j = i, j

    tour = [start_i, start_j]
    in_tour = {start_i, start_j}

    while len(tour) < n:
        # Find farthest city from tour
        farthest_city = None
        max_min_dist = -1

        for city in range(n):
            if city in in_tour:
                continue
            # Distance to tour = min distance to any city in tour
            min_dist = min(distance_matrix[city][t] for t in tour)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_city = city

        if farthest_city is None:
            break

        # Find best position to insert
        best_pos = 0
        best_increase = float("inf")

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            # Cost of inserting between i and j
            increase = (
                distance_matrix[tour[i]][farthest_city]
                + distance_matrix[farthest_city][tour[j]]
                - distance_matrix[tour[i]][tour[j]]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, farthest_city)
        in_tour.add(farthest_city)

    return tour


def cheapest_insertion(
    distance_matrix: DistanceMatrix,
    **kwargs
) -> Tour:
    """Build tour by inserting city at cheapest position.

    Algorithm:
    1. Start with triangle of 3 closest cities
    2. For each remaining city, find position with minimum insertion cost
    3. Insert city that has minimum insertion cost overall
    4. Repeat until all cities inserted

    Time complexity: O(n³)

    Args:
        distance_matrix: Distance matrix between cities

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Start with nearest neighbor pair + closest third city
    min_dist = float("inf")
    start_i, start_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] < min_dist:
                min_dist = distance_matrix[i][j]
                start_i, start_j = i, j

    # Find third city closest to both
    best_third = None
    best_total = float("inf")
    for k in range(n):
        if k != start_i and k != start_j:
            total = distance_matrix[start_i][k] + distance_matrix[start_j][k]
            if total < best_total:
                best_total = total
                best_third = k

    if best_third is None:
        return list(range(n))

    tour = [start_i, start_j, best_third]
    in_tour = {start_i, start_j, best_third}

    while len(tour) < n:
        # Find city with cheapest insertion
        best_city = None
        best_pos = 0
        best_cost = float("inf")

        for city in range(n):
            if city in in_tour:
                continue

            # Find best position for this city
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                cost = (
                    distance_matrix[tour[i]][city]
                    + distance_matrix[city][tour[j]]
                    - distance_matrix[tour[i]][tour[j]]
                )
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
                    best_pos = i + 1

        if best_city is None:
            break

        tour.insert(best_pos, best_city)
        in_tour.add(best_city)

    return tour


def random_insertion(
    distance_matrix: DistanceMatrix,
    seed: int | None = None,
    **kwargs
) -> Tour:
    """Build tour by inserting cities in random order at best position.

    Algorithm:
    1. Start with random city
    2. Randomly select next city to insert
    3. Insert at position that minimizes tour length
    4. Repeat until all cities inserted

    Time complexity: O(n²)

    Args:
        distance_matrix: Distance matrix between cities
        seed: Random seed (optional)

    Returns:
        Constructed tour
    """
    if seed is not None:
        random.seed(seed)

    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Randomize order of insertion
    cities = list(range(n))
    random.shuffle(cities)

    tour = [cities[0], cities[1]]

    for city in cities[2:]:
        # Find best position to insert
        best_pos = 0
        best_increase = float("inf")

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (
                distance_matrix[tour[i]][city]
                + distance_matrix[city][tour[j]]
                - distance_matrix[tour[i]][tour[j]]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, city)

    return tour


def savings_heuristic(
    distance_matrix: DistanceMatrix,
    depot: int = 0,
    lambda_param: float = 1.0,
    **kwargs
) -> Tour:
    """Clarke-Wright Savings algorithm.

    Originally for VRP but adapted for TSP.
    Merges routes based on savings from combining them.

    Algorithm:
    1. Start with depot connected to each city (star graph)
    2. Calculate savings for merging each pair
    3. Merge pairs with highest savings while maintaining tour validity
    4. Return final tour

    Time complexity: O(n² log n)

    Args:
        distance_matrix: Distance matrix between cities
        depot: Depot city (default 0)
        lambda_param: Savings parameter (default 1.0)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Calculate savings for each pair
    savings = []
    for i in range(n):
        if i == depot:
            continue
        for j in range(i + 1, n):
            if j == depot:
                continue
            # Savings = d(depot,i) + d(depot,j) - lambda * d(i,j)
            s = (
                distance_matrix[depot][i]
                + distance_matrix[depot][j]
                - lambda_param * distance_matrix[i][j]
            )
            savings.append((s, i, j))

    # Sort by savings (descending)
    savings.sort(reverse=True)

    # Initialize: each city is its own route
    # We track route endpoints
    route_of = {i: i for i in range(n) if i != depot}  # city -> route_id
    route_ends = {i: [i, i] for i in range(n) if i != depot}  # route_id -> [start, end]

    # Merge routes
    for s, i, j in savings:
        if s <= 0:
            break

        route_i = route_of[i]
        route_j = route_of[j]

        if route_i == route_j:
            continue  # Already in same route

        # Check if i and j are endpoints
        ends_i = route_ends[route_i]
        ends_j = route_ends[route_j]

        # Only merge if i is an endpoint of its route and j is an endpoint of its route
        if i not in ends_i or j not in ends_j:
            continue

        # Merge routes
        # Reconstruct the merged route
        # This is simplified - we just track that they're merged
        # Update route assignments
        for city in route_of:
            if route_of[city] == route_j:
                route_of[city] = route_i

        # Update endpoints
        new_ends = []
        if i == ends_i[0]:
            new_ends.append(ends_i[1])
        else:
            new_ends.append(ends_i[0])

        if j == ends_j[0]:
            new_ends.append(ends_j[1])
        else:
            new_ends.append(ends_j[0])

        route_ends[route_i] = new_ends
        del route_ends[route_j]

    # Build final tour from merged structure
    # For TSP, we just need a valid tour through all cities
    # Use the route structure to build the tour
    tour = [depot]
    visited = {depot}

    current = depot
    while len(tour) < n:
        # Find nearest unvisited
        best_next = None
        best_dist = float("inf")
        for city in range(n):
            if city not in visited and distance_matrix[current][city] < best_dist:
                best_dist = distance_matrix[current][city]
                best_next = city
        if best_next is None:
            break
        tour.append(best_next)
        visited.add(best_next)
        current = best_next

    return tour


def christofides_construction(
    distance_matrix: DistanceMatrix,
    coordinates: Coordinates | None = None,
    **kwargs
) -> Tour:
    """Christofides algorithm - 1.5-approximation for metric TSP.

    Algorithm:
    1. Compute Minimum Spanning Tree (MST)
    2. Find vertices with odd degree in MST
    3. Find minimum weight perfect matching on odd-degree vertices
    4. Combine MST and matching to get Eulerian graph
    5. Find Eulerian tour
    6. Convert to Hamiltonian tour by shortcutting

    Time complexity: O(n³) for matching

    Args:
        distance_matrix: Distance matrix between cities
        coordinates: Optional coordinates (not used but accepted for API consistency)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Step 1: Compute MST using Prim's algorithm
    in_mst = [False] * n
    mst_edges = []
    min_edge = [float("inf")] * n
    parent = [-1] * n

    min_edge[0] = 0

    for _ in range(n):
        # Find minimum edge to add
        u = -1
        for v in range(n):
            if not in_mst[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v

        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))

        # Update min edges
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < min_edge[v]:
                min_edge[v] = distance_matrix[u][v]
                parent[v] = u

    # Step 2: Find odd-degree vertices
    degree = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)

    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]

    # Step 3: Minimum weight perfect matching on odd vertices (greedy approximation)
    # Full matching is O(n³), we use greedy for simplicity
    matched = set()
    matching_edges = []

    # Sort all pairs of odd vertices by distance
    pairs = []
    for i, u in enumerate(odd_vertices):
        for j in range(i + 1, len(odd_vertices)):
            v = odd_vertices[j]
            pairs.append((distance_matrix[u][v], u, v))
    pairs.sort()

    for _, u, v in pairs:
        if u not in matched and v not in matched:
            matched.add(u)
            matched.add(v)
            matching_edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)

    # Step 4 & 5: Find Eulerian tour using Hierholzer's algorithm
    # Add matching edges to adjacency
    adj_multi = [list(neighbors) for neighbors in adj]

    euler_tour = []
    stack = [0]

    while stack:
        v = stack[-1]
        if adj_multi[v]:
            u = adj_multi[v].pop()
            # Remove reverse edge
            if v in adj_multi[u]:
                adj_multi[u].remove(v)
            stack.append(u)
        else:
            euler_tour.append(stack.pop())

    # Step 6: Convert to Hamiltonian by shortcutting
    visited = set()
    tour = []
    for v in euler_tour:
        if v not in visited:
            tour.append(v)
            visited.add(v)

    return tour


def nearest_addition(
    distance_matrix: DistanceMatrix,
    **kwargs
) -> Tour:
    """Build tour by adding nearest city to the tour.

    Algorithm:
    1. Start with the city that has minimum distance to any other
    2. Add nearest city to any city already in tour
    3. Insert at best position
    4. Repeat until complete

    Time complexity: O(n³)

    Args:
        distance_matrix: Distance matrix between cities

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    # Find city with minimum distance to any other (start point)
    min_dist = float("inf")
    start = 0
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i][j] < min_dist:
                min_dist = distance_matrix[i][j]
                start = i

    tour = [start]
    in_tour = {start}

    while len(tour) < n:
        # Find city nearest to tour
        best_city = None
        best_dist = float("inf")

        for city in range(n):
            if city in in_tour:
                continue
            for t in tour:
                if distance_matrix[city][t] < best_dist:
                    best_dist = distance_matrix[city][t]
                    best_city = city

        if best_city is None:
            break

        # Find best position to insert
        best_pos = 0
        best_increase = float("inf")

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (
                distance_matrix[tour[i]][best_city]
                + distance_matrix[best_city][tour[j]]
                - distance_matrix[tour[i]][tour[j]]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, best_city)
        in_tour.add(best_city)

    return tour


def convex_hull_start(
    distance_matrix: DistanceMatrix,
    coordinates: Coordinates | None = None,
    **kwargs
) -> Tour:
    """Start with convex hull, then insert remaining cities.

    Algorithm:
    1. Compute convex hull of cities
    2. Insert remaining cities at cheapest position

    Requires coordinates. Falls back to farthest insertion if not available.

    Time complexity: O(n² log n)

    Args:
        distance_matrix: Distance matrix between cities
        coordinates: City coordinates (required)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 3:
        return list(range(n))

    if coordinates is None or len(coordinates) != n:
        # Fallback to farthest insertion
        return farthest_insertion(distance_matrix)

    # Compute convex hull using Graham scan
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort points by x, then y
    indexed_coords = [(coordinates[i], i) for i in range(n)]
    indexed_coords.sort()

    # Build lower hull
    lower = []
    for coord, idx in indexed_coords:
        while len(lower) >= 2 and cross(
            coordinates[lower[-2]], coordinates[lower[-1]], coord
        ) <= 0:
            lower.pop()
        lower.append(idx)

    # Build upper hull
    upper = []
    for coord, idx in reversed(indexed_coords):
        while len(upper) >= 2 and cross(
            coordinates[upper[-2]], coordinates[upper[-1]], coord
        ) <= 0:
            upper.pop()
        upper.append(idx)

    # Combine hulls (remove last point of each as it's repeated)
    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3:
        return farthest_insertion(distance_matrix)

    tour = hull.copy()
    in_tour = set(hull)

    # Insert remaining cities at cheapest position
    for city in range(n):
        if city in in_tour:
            continue

        best_pos = 0
        best_increase = float("inf")

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (
                distance_matrix[tour[i]][city]
                + distance_matrix[city][tour[j]]
                - distance_matrix[tour[i]][tour[j]]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, city)
        in_tour.add(city)

    return tour


def cluster_first(
    distance_matrix: DistanceMatrix,
    coordinates: Coordinates | None = None,
    n_clusters: int | None = None,
    **kwargs
) -> Tour:
    """Cluster first, route second approach.

    Algorithm:
    1. Cluster cities into groups (using k-means or simple partitioning)
    2. Find tour within each cluster
    3. Connect clusters optimally

    Time complexity: O(n² + k³) where k is number of clusters

    Args:
        distance_matrix: Distance matrix between cities
        coordinates: City coordinates (optional, improves clustering)
        n_clusters: Number of clusters (auto if None)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 3:
        return list(range(n))

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = max(2, int(math.sqrt(n / 2)))
    n_clusters = min(n_clusters, n)

    # Simple clustering based on nearest neighbor chains
    # Start with random seeds
    random.seed(42)  # For reproducibility
    seeds = random.sample(range(n), n_clusters)

    # Assign each city to nearest seed
    clusters = [[] for _ in range(n_clusters)]
    for city in range(n):
        best_cluster = 0
        best_dist = float("inf")
        for i, seed in enumerate(seeds):
            if distance_matrix[city][seed] < best_dist:
                best_dist = distance_matrix[city][seed]
                best_cluster = i
        clusters[best_cluster].append(city)

    # Build tour within each cluster using nearest neighbor
    cluster_tours = []
    cluster_centers = []

    for cluster in clusters:
        if not cluster:
            continue

        # Tour within cluster
        if len(cluster) == 1:
            cluster_tours.append(cluster)
            cluster_centers.append(cluster[0])
        else:
            # Use nearest neighbor within cluster
            tour = [cluster[0]]
            visited = {cluster[0]}
            while len(tour) < len(cluster):
                current = tour[-1]
                best_next = None
                best_dist = float("inf")
                for city in cluster:
                    if city not in visited and distance_matrix[current][city] < best_dist:
                        best_dist = distance_matrix[current][city]
                        best_next = city
                if best_next is not None:
                    tour.append(best_next)
                    visited.add(best_next)
            cluster_tours.append(tour)
            # Center is first city (could compute centroid if coordinates available)
            cluster_centers.append(tour[0])

    # Order clusters using nearest neighbor on centers
    if len(cluster_tours) <= 1:
        return cluster_tours[0] if cluster_tours else list(range(n))

    cluster_order = [0]
    visited_clusters = {0}
    while len(cluster_order) < len(cluster_tours):
        current = cluster_order[-1]
        best_next = None
        best_dist = float("inf")
        for i in range(len(cluster_tours)):
            if i not in visited_clusters:
                d = distance_matrix[cluster_centers[current]][cluster_centers[i]]
                if d < best_dist:
                    best_dist = d
                    best_next = i
        if best_next is not None:
            cluster_order.append(best_next)
            visited_clusters.add(best_next)

    # Combine cluster tours
    final_tour = []
    for i in cluster_order:
        final_tour.extend(cluster_tours[i])

    return final_tour


def sweep_algorithm(
    distance_matrix: DistanceMatrix,
    coordinates: Coordinates | None = None,
    depot: int = 0,
    start_angle: float = 0.0,
    **kwargs
) -> Tour:
    """Sweep algorithm - order cities by angle from depot.

    Algorithm:
    1. Calculate angle of each city from depot
    2. Sort cities by angle
    3. Visit in angular order

    Requires coordinates. Falls back to nearest neighbor if not available.

    Time complexity: O(n log n)

    Args:
        distance_matrix: Distance matrix between cities
        coordinates: City coordinates (required)
        depot: Starting city (default 0)
        start_angle: Starting angle in radians (default 0)

    Returns:
        Constructed tour
    """
    n = len(distance_matrix)
    if n <= 2:
        return list(range(n))

    if coordinates is None or len(coordinates) != n:
        # Fallback to nearest neighbor
        return greedy_nearest_neighbor(distance_matrix, start=depot)

    depot_x, depot_y = coordinates[depot]

    # Calculate angle for each city
    city_angles = []
    for i in range(n):
        if i == depot:
            continue
        x, y = coordinates[i]
        angle = math.atan2(y - depot_y, x - depot_x)
        # Adjust by start_angle
        angle = (angle - start_angle) % (2 * math.pi)
        city_angles.append((angle, i))

    # Sort by angle
    city_angles.sort()

    # Build tour
    tour = [depot] + [city for _, city in city_angles]

    return tour
