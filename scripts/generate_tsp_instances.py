#!/usr/bin/env python3
"""Generate diverse TSP instances with optimal solutions using pyconcorde.

Instance types for capturing different permutation problem topologies:
- uniform: Random uniform distribution (baseline)
- clustered: Cities grouped in clusters (tests inter-cluster decisions)
- grid: Regular grid with noise (tests local structure exploitation)
- concentric: Concentric circles (tests radial patterns)
- mixed: Combination of patterns (tests adaptivity)

Usage:
    # Generate diverse instances (recommended)
    uv run python scripts/generate_tsp_instances.py --size 50 --diverse

    # Generate specific type
    uv run python scripts/generate_tsp_instances.py --size 100 --type clustered --count 3

    # Generate all types for multiple sizes
    uv run python scripts/generate_tsp_instances.py --all-sizes --diverse
"""

import argparse
from pathlib import Path

import numpy as np
from concorde.tsp import TSPSolver


INSTANCE_TYPES = ["uniform", "clustered", "grid", "concentric", "mixed"]


def generate_uniform_coords(n: int, rng: np.random.Generator, scale: float = 1000.0) -> np.ndarray:
    """Random uniform distribution - baseline."""
    return rng.uniform(0, scale, size=(n, 2))


def generate_clustered_coords(n: int, rng: np.random.Generator, scale: float = 1000.0) -> np.ndarray:
    """Clustered cities - tests greedy inter-cluster jumps."""
    n_clusters = max(3, n // 15)  # ~15 cities per cluster
    cluster_centers = rng.uniform(scale * 0.1, scale * 0.9, size=(n_clusters, 2))
    cluster_std = scale * 0.08  # Tight clusters

    coords = []
    for i in range(n):
        center = cluster_centers[i % n_clusters]
        point = rng.normal(center, cluster_std)
        point = np.clip(point, 0, scale)
        coords.append(point)

    return np.array(coords)


def generate_grid_coords(n: int, rng: np.random.Generator, scale: float = 1000.0) -> np.ndarray:
    """Grid with noise - tests local structure exploitation."""
    side = int(np.ceil(np.sqrt(n)))
    spacing = scale / (side + 1)
    noise_std = spacing * 0.15  # Small noise

    coords = []
    for i in range(n):
        row, col = i // side, i % side
        x = (col + 1) * spacing + rng.normal(0, noise_std)
        y = (row + 1) * spacing + rng.normal(0, noise_std)
        coords.append([np.clip(x, 0, scale), np.clip(y, 0, scale)])

    return np.array(coords)


def generate_concentric_coords(n: int, rng: np.random.Generator, scale: float = 1000.0) -> np.ndarray:
    """Concentric circles - tests radial/savings patterns."""
    center = scale / 2
    n_rings = max(3, n // 20)
    ring_radii = np.linspace(scale * 0.1, scale * 0.45, n_rings)

    coords = []
    for i in range(n):
        ring = i % n_rings
        radius = ring_radii[ring] + rng.normal(0, scale * 0.02)
        angle = rng.uniform(0, 2 * np.pi)
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        coords.append([np.clip(x, 0, scale), np.clip(y, 0, scale)])

    return np.array(coords)


def generate_mixed_coords(n: int, rng: np.random.Generator, scale: float = 1000.0) -> np.ndarray:
    """Mixed patterns - tests operator adaptivity."""
    # 40% clustered, 30% uniform, 30% grid-like
    n_clustered = int(n * 0.4)
    n_uniform = int(n * 0.3)
    n_grid = n - n_clustered - n_uniform

    coords = []

    # Clustered region (top-left quadrant)
    n_clusters = max(2, n_clustered // 10)
    cluster_centers = rng.uniform(0, scale * 0.45, size=(n_clusters, 2))
    for i in range(n_clustered):
        center = cluster_centers[i % n_clusters]
        point = rng.normal(center, scale * 0.05)
        coords.append(np.clip(point, 0, scale * 0.5))

    # Uniform region (bottom-right quadrant)
    for _ in range(n_uniform):
        point = rng.uniform([scale * 0.5, scale * 0.5], [scale, scale])
        coords.append(point)

    # Grid region (top-right quadrant)
    side = int(np.ceil(np.sqrt(n_grid)))
    spacing = (scale * 0.45) / (side + 1)
    for i in range(n_grid):
        row, col = i // side, i % side
        x = scale * 0.55 + (col + 1) * spacing + rng.normal(0, spacing * 0.1)
        y = (row + 1) * spacing + rng.normal(0, spacing * 0.1)
        coords.append([np.clip(x, scale * 0.5, scale), np.clip(y, 0, scale * 0.5)])

    return np.array(coords)


def generate_coords(n: int, instance_type: str, seed: int, scale: float = 1000.0) -> np.ndarray:
    """Generate coordinates based on instance type."""
    rng = np.random.default_rng(seed)

    generators = {
        "uniform": generate_uniform_coords,
        "clustered": generate_clustered_coords,
        "grid": generate_grid_coords,
        "concentric": generate_concentric_coords,
        "mixed": generate_mixed_coords,
    }

    if instance_type not in generators:
        raise ValueError(f"Unknown type: {instance_type}. Choose from {INSTANCE_TYPES}")

    return generators[instance_type](n, rng, scale)


def compute_optimal(coords: np.ndarray) -> tuple[float, list[int]]:
    """Compute optimal tour using Concorde."""
    solver = TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D")
    solution = solver.solve()
    return solution.optimal_value, list(solution.tour)


def save_tsp_file(path: Path, name: str, coords: np.ndarray, optimal: float, instance_type: str):
    """Save instance in TSPLIB format."""
    n = len(coords)
    type_descriptions = {
        "uniform": "Random uniform distribution",
        "clustered": "Clustered cities (like berlin52)",
        "grid": "Grid with noise (like PCB drilling)",
        "concentric": "Concentric rings pattern",
        "mixed": "Mixed patterns (tests adaptivity)",
    }
    desc = type_descriptions.get(instance_type, "Generated instance")

    with open(path, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: {desc}, {n} cities, optimal={optimal:.0f}\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse TSP instances with optimal solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instance types:
  uniform     Random uniform distribution (baseline)
  clustered   Cities in clusters (like real cities, berlin52)
  grid        Regular grid with noise (like PCB drilling, pr2392)
  concentric  Concentric circles (radial patterns)
  mixed       Combination of patterns (tests adaptivity)

Examples:
  # Generate one instance of each type for size 50
  uv run python scripts/generate_tsp_instances.py --size 50 --diverse

  # Generate 3 clustered instances of size 100
  uv run python scripts/generate_tsp_instances.py --size 100 --type clustered --count 3

  # Generate diverse instances for all standard sizes
  uv run python scripts/generate_tsp_instances.py --all-sizes --diverse
        """,
    )
    parser.add_argument("--size", type=int, help="Number of cities")
    parser.add_argument("--count", type=int, default=1, help="Instances per type (default: 1)")
    parser.add_argument("--type", choices=INSTANCE_TYPES, help="Instance type (default: all if --diverse)")
    parser.add_argument("--diverse", action="store_true", help="Generate one of each type")
    parser.add_argument("--all-sizes", action="store_true", help="Generate for sizes 50, 100, 150")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", default="data/instances/tsp_diverse", help="Output directory")
    args = parser.parse_args()

    # Determine sizes to generate
    if args.all_sizes:
        sizes = [50, 100, 150]
    elif args.size:
        sizes = [args.size]
    else:
        parser.error("Must specify --size or --all-sizes")

    # Determine types to generate
    if args.diverse:
        types = INSTANCE_TYPES
    elif args.type:
        types = [args.type]
    else:
        types = ["uniform"]  # Default to uniform for backwards compatibility

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(sizes) * len(types) * args.count
    print(f"Generating {total} TSP instances...")
    print(f"  Sizes: {sizes}")
    print(f"  Types: {types}")
    print(f"  Count per type: {args.count}")
    print()

    generated = 0
    for size in sizes:
        for instance_type in types:
            for i in range(1, args.count + 1):
                seed = args.seed + hash((size, instance_type, i)) % 10000
                name = f"tsp{size}_{instance_type}_{i:02d}"

                print(f"  {name}: generating...", end=" ", flush=True)

                coords = generate_coords(size, instance_type, seed)

                print("solving...", end=" ", flush=True)
                optimal, tour = compute_optimal(coords)

                path = output_dir / f"{name}.tsp"
                save_tsp_file(path, name, coords, optimal, instance_type)

                print(f"optimal={optimal:.0f} ✓")
                generated += 1

    print(f"\nDone! Generated {generated} instances in {output_dir}")


if __name__ == "__main__":
    main()
