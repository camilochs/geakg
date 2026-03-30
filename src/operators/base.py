"""Base utilities for TSP operators."""

import math
import random
from typing import TypeAlias

# Type aliases
Tour: TypeAlias = list[int]
DistanceMatrix: TypeAlias = list[list[float]]
Coordinates: TypeAlias = list[tuple[float, float]]


def calculate_tour_cost(tour: Tour, distance_matrix: DistanceMatrix) -> float:
    """Calculate total tour cost.

    Args:
        tour: List of city indices representing the tour
        distance_matrix: Distance matrix between cities

    Returns:
        Total tour cost
    """
    n = len(tour)
    return sum(
        distance_matrix[tour[i]][tour[(i + 1) % n]]
        for i in range(n)
    )


def calculate_segment_cost(
    tour: Tour,
    start: int,
    end: int,
    distance_matrix: DistanceMatrix
) -> float:
    """Calculate cost of a segment of the tour.

    Args:
        tour: The tour
        start: Start index (inclusive)
        end: End index (inclusive)
        distance_matrix: Distance matrix

    Returns:
        Segment cost
    """
    cost = 0.0
    for i in range(start, end):
        cost += distance_matrix[tour[i]][tour[i + 1]]
    return cost


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def is_valid_tour(tour: Tour, n_cities: int) -> bool:
    """Check if tour is valid (visits each city exactly once).

    Args:
        tour: The tour to validate
        n_cities: Expected number of cities

    Returns:
        True if valid
    """
    if len(tour) != n_cities:
        return False
    if set(tour) != set(range(n_cities)):
        return False
    return True


def reverse_segment(tour: Tour, i: int, j: int) -> Tour:
    """Reverse segment of tour between indices i and j (inclusive).

    Args:
        tour: The tour
        i: Start index
        j: End index

    Returns:
        New tour with reversed segment
    """
    new_tour = tour.copy()
    new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour


def rotate_tour(tour: Tour, start_city: int) -> Tour:
    """Rotate tour so it starts with a specific city.

    Args:
        tour: The tour
        start_city: City to start with

    Returns:
        Rotated tour
    """
    if start_city not in tour:
        return tour
    idx = tour.index(start_city)
    return tour[idx:] + tour[:idx]


def get_edges(tour: Tour) -> set[tuple[int, int]]:
    """Get set of edges in tour (undirected).

    Args:
        tour: The tour

    Returns:
        Set of (min, max) tuples representing edges
    """
    n = len(tour)
    edges = set()
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        edges.add((min(a, b), max(a, b)))
    return edges


def create_distance_matrix(coordinates: Coordinates) -> DistanceMatrix:
    """Create distance matrix from coordinates.

    Args:
        coordinates: List of (x, y) tuples

    Returns:
        Distance matrix
    """
    n = len(coordinates)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(coordinates[i], coordinates[j])
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix


def get_nearest_neighbors(
    city: int,
    distance_matrix: DistanceMatrix,
    k: int | None = None,
    exclude: set[int] | None = None
) -> list[int]:
    """Get k nearest neighbors of a city.

    Args:
        city: City index
        distance_matrix: Distance matrix
        k: Number of neighbors (None for all)
        exclude: Cities to exclude

    Returns:
        List of neighbor indices sorted by distance
    """
    n = len(distance_matrix)
    exclude = exclude or set()
    exclude.add(city)

    neighbors = [
        (distance_matrix[city][j], j)
        for j in range(n) if j not in exclude
    ]
    neighbors.sort()

    result = [j for _, j in neighbors]
    if k is not None:
        result = result[:k]

    return result
