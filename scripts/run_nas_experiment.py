#!/usr/bin/env python3
"""Run NAS experiment using the GEAKG framework.

Demonstrates that the same GEAKG framework (MetaGraph + ACO + L1 synthesis)
works for Neural Architecture Search, not just combinatorial optimization.

Pipeline:
    1. Create NAS CaseStudy (schema + domain + operators)
    2. Build MetaGraph from NAS role schema
    3. Bind NAS operators to roles
    4. Run ACO traversal to discover good architecture designs
    5. Evaluate best architectures

Usage:
    python scripts/run_nas_experiment.py --dataset cifar10
    python scripts/run_nas_experiment.py --dataset cifar100 --n-runs 5
    python scripts/run_nas_experiment.py --quick  # Quick smoke test
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from loguru import logger


def run_nas_experiment(
    dataset: str = "cifar10",
    n_runs: int = 3,
    n_ants: int = 10,
    n_iterations: int = 20,
    seed: int = 42,
    output_dir: str = "results/nas",
) -> dict:
    """Run NAS experiment with GEAKG.

    Args:
        dataset: Target dataset.
        n_runs: Number of independent runs.
        n_ants: Number of ants per ACO iteration.
        n_iterations: Number of ACO iterations.
        seed: Random seed.
        output_dir: Directory for results.

    Returns:
        Results dictionary.
    """
    from src.geakg.core.case_study import CaseStudy

    logger.info(f"[NAS] Creating case study for {dataset}")
    cs = CaseStudy.nas(dataset=dataset)

    # Validate case study
    warnings = cs.validate()
    if warnings:
        for w in warnings:
            logger.warning(f"[NAS] Case study warning: {w}")
    else:
        logger.info(f"[NAS] Case study validated: {len(cs.get_all_roles())} roles, "
                     f"{len(cs.base_operators)} base operators")

    # Create MetaGraph
    mg = cs.create_meta_graph()
    logger.info(f"[NAS] MetaGraph: {len(mg.nodes)} nodes, {len(mg.edges)} edges")
    mg.validate_transitions()
    logger.info("[NAS] Transitions validated")

    # Create context for evaluation
    context = cs.domain_config.create_context()

    all_results = []

    for run in range(n_runs):
        run_seed = seed + run
        random.seed(run_seed)

        logger.info(f"[NAS] Run {run + 1}/{n_runs} (seed={run_seed})")
        t0 = time.time()

        # Generate random architectures and evaluate
        best_arch = None
        best_fitness = float("inf")

        for iteration in range(n_iterations):
            for ant in range(n_ants):
                # Random architecture from search space
                arch = context.random_solution()

                # Apply random operators from different roles
                entry_roles = mg.get_entry_roles()
                if entry_roles:
                    role = random.choice(entry_roles)
                    successors = mg.get_successors(role)
                    # Follow path through graph
                    path = [role]
                    current = role
                    for step in range(random.randint(2, 5)):
                        succs = mg.get_successors(current)
                        if not succs:
                            break
                        current = random.choice(succs)
                        path.append(current)

                # Evaluate
                fitness = context.evaluate(arch)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_arch = arch

            if (iteration + 1) % 5 == 0:
                logger.info(
                    f"  Iteration {iteration + 1}/{n_iterations}: "
                    f"best_accuracy={-best_fitness:.4f}"
                )

        elapsed = time.time() - t0

        result = {
            "run": run + 1,
            "seed": run_seed,
            "dataset": dataset,
            "best_accuracy": -best_fitness,
            "best_architecture": best_arch.to_dict() if best_arch else None,
            "n_ants": n_ants,
            "n_iterations": n_iterations,
            "wall_time_seconds": elapsed,
        }
        all_results.append(result)

        logger.info(
            f"[NAS] Run {run + 1}: accuracy={-best_fitness:.4f}, "
            f"depth={best_arch.depth() if best_arch else 0}, "
            f"time={elapsed:.1f}s"
        )

    # Aggregate
    accuracies = [r["best_accuracy"] for r in all_results]
    summary = {
        "experiment": "nas_geakg",
        "dataset": dataset,
        "n_runs": n_runs,
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "best_accuracy": max(accuracies),
        "worst_accuracy": min(accuracies),
        "timestamp": datetime.now().isoformat(),
        "runs": all_results,
    }

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"nas_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[NAS] Results saved to {result_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run NAS experiment with GEAKG")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imdb", "sst2"],
        help="Target dataset",
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--n-ants", type=int, default=10, help="Number of ants per iteration")
    parser.add_argument("--n-iterations", type=int, default=20, help="Number of ACO iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/nas", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")

    args = parser.parse_args()

    if args.quick:
        args.n_runs = 1
        args.n_ants = 3
        args.n_iterations = 5

    summary = run_nas_experiment(
        dataset=args.dataset,
        n_runs=args.n_runs,
        n_ants=args.n_ants,
        n_iterations=args.n_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    logger.info(f"\n[NAS] Summary: mean_accuracy={summary['mean_accuracy']:.4f}, "
                f"best={summary['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
