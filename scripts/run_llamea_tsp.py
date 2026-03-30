#!/usr/bin/env python3
"""Run LLaMEA best code on TSP instances.

Usage:
    uv run python scripts/run_llamea_tsp.py data/instances/tsp/pr226.tsp -t 226
    uv run python scripts/run_llamea_tsp.py data/instances/tsp/pcb442.tsp -t 442
    uv run python scripts/run_llamea_tsp.py data/instances/tsp/berlin52.tsp -t 60 --snapshot experiments/llamea_10k_gpt4omini.json
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path


def load_tsplib(filepath: str) -> tuple[list[list[float]], float | None]:
    """Load TSP instance from TSPLIB format."""
    with open(filepath) as f:
        lines = f.read().strip().split("\n")

    coords = []
    reading_coords = False
    dimension = 0
    optimal = None

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
        elif line.startswith("EOF") or line.startswith("EDGE_WEIGHT_SECTION"):
            break
        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    n = len(coords)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dm[i][j] = math.sqrt(dx * dx + dy * dy)

    return dm, optimal


# Default LLaMEA best code (gpt-4o-mini, 10k tokens)
DEFAULT_CODE = '''
def solve_tsp(distance_matrix: list[list[float]]) -> list[int]:
    import random

    def calculate_tour_length(tour):
        return sum(distance_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    def two_opt(tour):
        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                    if calculate_tour_length(new_tour) < calculate_tour_length(tour):
                        tour = new_tour
                        improved = True
        return tour

    n = len(distance_matrix)
    best_tour = None
    best_length = float('inf')

    for _ in range(100):
        start_city = random.randint(0, n - 1)
        tour = [start_city]
        visited = {start_city}

        current_city = start_city
        while len(tour) < n:
            next_city = min((distance_matrix[current_city][j], j) for j in range(n) if j not in visited)[1]
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city

        tour = two_opt(tour)
        tour_length = calculate_tour_length(tour)

        if tour_length < best_length:
            best_length = tour_length
            best_tour = tour

    return best_tour
'''


def load_code_from_snapshot(snapshot_path: str) -> str:
    """Load best_code from LLaMEA snapshot."""
    with open(snapshot_path) as f:
        data = json.load(f)
    return data.get("best_code", DEFAULT_CODE)


def run_with_timeout(solve_fn, dm, timeout_seconds):
    """Run solve_tsp with restarts until timeout."""
    start = time.time()
    best_cost = float('inf')
    best_tour = None
    restarts = 0
    n = len(dm)

    def calc_cost(tour):
        return sum(dm[tour[i]][tour[(i + 1) % n]] for i in range(n))

    print(f"\nRunning for {timeout_seconds}s...")

    while time.time() - start < timeout_seconds:
        try:
            tour = solve_fn(dm)
            cost = calc_cost(tour)
            restarts += 1

            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                elapsed = time.time() - start
                print(f"  [{elapsed:6.1f}s] Restart #{restarts}: {cost:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
            restarts += 1

    return best_tour, best_cost, restarts


def main():
    parser = argparse.ArgumentParser(description="Run LLaMEA best code on TSP")
    parser.add_argument("instance", help="Path to .tsp file")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--snapshot", help="Path to LLaMEA snapshot JSON (optional)")
    args = parser.parse_args()

    # Load code
    if args.snapshot:
        print(f"Loading code from: {args.snapshot}")
        code = load_code_from_snapshot(args.snapshot)
    else:
        print("Using default LLaMEA code (gpt-4o-mini)")
        code = DEFAULT_CODE

    # Compile code
    exec_globals = {
        "__builtins__": __builtins__,
        "random": random,
        "math": math,
    }
    local_vars = {}
    exec(code, exec_globals, local_vars)
    solve_fn = local_vars["solve_tsp"]

    # Load instance
    print(f"Loading instance: {args.instance}")
    dm, optimal = load_tsplib(args.instance)
    n = len(dm)
    print(f"Instance: n={n}, optimal={optimal}")

    # Run
    tour, cost, restarts = run_with_timeout(solve_fn, dm, args.timeout)

    # Results
    print(f"\n{'='*50}")
    print(f"Restarts: {restarts}")
    print(f"Best cost: {cost:.2f}")
    if optimal:
        gap = 100 * (cost - optimal) / optimal
        print(f"Gap: {gap:.2f}%")


if __name__ == "__main__":
    main()
