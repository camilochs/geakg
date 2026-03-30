#!/usr/bin/env python3
"""Run optimization using a pre-generated L0 operator pool.

This script runs ACO optimization WITHOUT any LLM calls at runtime,
using operators from a pre-generated L0 pool.

Usage:
    # Run with pool on single instance
    uv run python scripts/run_with_pool.py \
        --pool pools/tsp_pool.json \
        --instance data/instances/tsp/berlin52.tsp

    # Run with pool on multiple instances
    uv run python scripts/run_with_pool.py \
        --pool pools/tsp_pool.json \
        --instances data/instances/tsp/*.tsp \
        --n-runs 5

    # Compare with synthesis runtime synthesis
    uv run python scripts/run_with_pool.py \
        --pool pools/tsp_pool.json \
        --instance data/instances/tsp/berlin52.tsp \
        --compare-synth
"""

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

from loguru import logger


def run_with_pool(instance_path: str, pool_path: str, seed: int = 42) -> dict:
    """Run optimization with L0 pool (no LLM at runtime).

    Args:
        instance_path: Path to TSP instance
        pool_path: Path to operator pool JSON
        seed: Random seed

    Returns:
        Result dictionary with metrics
    """
    random.seed(seed)

    from src.geakg.pipeline import NSGGEPipeline, LLMBackend
    from src.domains.tsp import TSPDomain

    # Load instance
    domain = TSPDomain()
    tsp_instance = domain.load_instance(Path(instance_path))

    logger.info(f"Instance: {tsp_instance.name} (n={tsp_instance.dimension})")

    # Create pipeline with pool (no LLM needed for runtime)
    pipeline = NSGGEPipeline(
        backend=LLMBackend.OLLAMA,  # Dummy, won't be used
        enable_synthesis=False,  # No synthesis synthesis
        operator_pool_path=pool_path,
    )

    # Generate with predefined metagraph and pool
    result = pipeline.generate(
        domain="tsp",
        use_predefined_metagraph=True,  # No LLM for metagraph either
        operator_pool_path=pool_path,
    )

    logger.info(f"Pool loaded: {result.generation_stats.get('pool_operators', 0)} operators")

    # Create fitness function
    from src.geakg.contexts.tsp import TSPContext

    ctx = TSPContext(tsp_instance.distance_matrix)

    def fitness_fn(solution: list) -> float:
        return ctx.evaluate(solution)

    # Run ACO
    start_time = time.time()

    from src.geakg.aco import AntColonyOptimizer

    aco = AntColonyOptimizer(
        selector=result.selector,
        n_ants=15,
        n_iterations=100,
        fitness_fn=fitness_fn,
        instance_size=tsp_instance.dimension,
    )

    # Initial solution
    initial = list(range(tsp_instance.dimension))
    random.shuffle(initial)

    best_solution, best_fitness = aco.run(initial)
    elapsed = time.time() - start_time

    # Calculate gap if optimal known
    gap = None
    if tsp_instance.optimal_cost:
        gap = (best_fitness - tsp_instance.optimal_cost) / tsp_instance.optimal_cost * 100

    return {
        "instance": tsp_instance.name,
        "dimension": tsp_instance.dimension,
        "best_fitness": best_fitness,
        "optimal": tsp_instance.optimal_cost,
        "gap_percent": gap,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "pool_operators": result.generation_stats.get("pool_operators", 0),
        "llm_tokens_runtime": 0,  # No LLM calls
    }


def run_with_synth(instance_path: str, seed: int = 42) -> dict:
    """Run optimization with synthesis synthesis (LLM at runtime).

    For comparison purposes.
    """
    # This would require an actual LLM setup
    # Placeholder for comparison
    return {
        "error": "synthesis comparison requires LLM setup",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run optimization with L0 operator pool (no LLM at runtime)"
    )
    parser.add_argument(
        "--pool",
        type=str,
        required=True,
        help="Path to L0 operator pool JSON",
    )
    parser.add_argument(
        "--instance",
        type=str,
        help="Single instance file path",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        help="Multiple instance files or glob patterns",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of runs per instance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Collect instances
    instance_paths = []
    if args.instance:
        instance_paths.append(args.instance)
    if args.instances:
        for pattern in args.instances:
            paths = glob.glob(pattern)
            if not paths:
                if Path(pattern).exists():
                    paths = [pattern]
            instance_paths.extend(paths)

    if not instance_paths:
        logger.error("No instances specified")
        sys.exit(1)

    # Check pool exists
    if not Path(args.pool).exists():
        logger.error(f"Pool not found: {args.pool}")
        sys.exit(1)

    logger.info(f"Running with pool: {args.pool}")
    logger.info(f"Instances: {len(instance_paths)}")
    logger.info(f"Runs per instance: {args.n_runs}")

    # Run experiments
    all_results = []

    for instance_path in instance_paths:
        for run in range(args.n_runs):
            seed = args.seed + run
            logger.info(f"\n{'='*60}")
            logger.info(f"Instance: {instance_path}, Run: {run+1}/{args.n_runs}")

            try:
                result = run_with_pool(instance_path, args.pool, seed)
                result["run"] = run + 1
                all_results.append(result)

                if result.get("gap_percent") is not None:
                    logger.info(
                        f"Result: {result['best_fitness']:.2f} "
                        f"(gap: {result['gap_percent']:.2f}%) "
                        f"in {result['elapsed_seconds']:.2f}s"
                    )
                else:
                    logger.info(
                        f"Result: {result['best_fitness']:.2f} "
                        f"in {result['elapsed_seconds']:.2f}s"
                    )
            except Exception as e:
                logger.error(f"Run failed: {e}")
                all_results.append({
                    "instance": instance_path,
                    "run": run + 1,
                    "error": str(e),
                })

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    successful = [r for r in all_results if "error" not in r]
    if successful:
        avg_gap = sum(r.get("gap_percent", 0) or 0 for r in successful) / len(successful)
        avg_time = sum(r["elapsed_seconds"] for r in successful) / len(successful)

        print(f"Successful runs: {len(successful)}/{len(all_results)}")
        print(f"Average gap: {avg_gap:.2f}%")
        print(f"Average time: {avg_time:.2f}s")
        print(f"LLM tokens (runtime): 0")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
