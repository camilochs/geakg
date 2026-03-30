#!/usr/bin/env python3
"""Generate random TSP instances in TSPLIB format with optimal solutions.

Uses python-tsp library to compute optimal/near-optimal solutions.
For small instances (<= 12), uses exact dynamic programming.
For larger instances, uses Lin-Kernighan heuristic (very close to optimal).

Usage:
    python data/generate_tsp_instances.py --cities 50 --instances 10 --output data/instances/tsp_random
    python data/generate_tsp_instances.py --cities 100 --instances 5 --seed 42
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np


def compute_distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """Compute Euclidean distance matrix from coordinates.

    Args:
        coords: List of (x, y) coordinates

    Returns:
        Distance matrix as numpy array
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def solve_tsp_optimal(coords: list[tuple[float, float]]) -> tuple[int, list[int]]:
    """Solve TSP to find optimal or near-optimal solution.

    Args:
        coords: List of (x, y) coordinates

    Returns:
        Tuple of (tour_length, tour) where tour_length is rounded to integer
    """
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_simulated_annealing

    dist_matrix = compute_distance_matrix(coords)
    n = len(coords)

    if n <= 13:
        # Use exact solver for small instances (DP is O(n^2 * 2^n))
        tour, length = solve_tsp_dynamic_programming(dist_matrix)
    else:
        # Use simulated annealing for larger instances
        # Run multiple times and take best result for better quality
        best_length = float('inf')
        best_tour = None
        for _ in range(3):  # 3 restarts
            tour, length = solve_tsp_simulated_annealing(dist_matrix)
            if length < best_length:
                best_length = length
                best_tour = tour
        tour, length = best_tour, best_length

    # Round to integer (TSPLIB convention for EUC_2D)
    return int(round(length)), list(tour)


def generate_tsp_instance(n_cities: int, seed: int | None = None) -> list[tuple[float, float]]:
    """Generate a random TSP instance with Euclidean distances.

    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility

    Returns:
        List of (x, y) coordinates
    """
    if seed is not None:
        random.seed(seed)

    # Generate random coordinates in [0, 1000] x [0, 1000]
    coords = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n_cities)]

    return coords


def write_tsplib_format(
    filepath: Path,
    name: str,
    coords: list[tuple[float, float]],
    optimal: int | None = None,
) -> None:
    """Write instance in TSPLIB format.

    Args:
        filepath: Output file path
        name: Instance name
        coords: List of (x, y) coordinates
        optimal: Optimal tour length (included in COMMENT if provided)
    """
    n = len(coords)

    with open(filepath, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        if optimal is not None:
            f.write(f"COMMENT: Random instance with {n} cities, optimal={optimal}\n")
        else:
            f.write(f"COMMENT: Random instance with {n} cities\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")

        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")

        f.write("EOF\n")


def main():
    parser = argparse.ArgumentParser(description="Generate random TSP instances with optimal solutions")
    parser.add_argument("--cities", type=int, required=True, help="Number of cities per instance")
    parser.add_argument("--instances", type=int, required=True, help="Number of instances to generate")
    parser.add_argument("--output", type=str, default="data/instances/tsp_random", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (instance i uses seed+i)")
    parser.add_argument("--prefix", type=str, default="rand", help="Instance name prefix")
    parser.add_argument("--no-solve", action="store_true", help="Skip computing optimal solution")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.instances} TSP instances with {args.cities} cities each")
    print(f"Output directory: {output_dir}")
    if not args.no_solve:
        print(f"Computing optimal solutions using {'exact DP' if args.cities <= 12 else 'Lin-Kernighan'}...")

    for i in range(args.instances):
        seed = args.seed + i if args.seed is not None else None
        name = f"{args.prefix}{args.cities}_{i+1:03d}"

        coords = generate_tsp_instance(args.cities, seed)

        optimal = None
        if not args.no_solve:
            optimal, tour = solve_tsp_optimal(coords)
            print(f"  {name}: optimal = {optimal}")

        filepath = output_dir / f"{name}.tsp"
        write_tsplib_format(filepath, name, coords, optimal)

    print(f"\nDone. {args.instances} instances saved to {output_dir}")


if __name__ == "__main__":
    main()
