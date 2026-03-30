#!/usr/bin/env python3
"""LLaMEA baseline experiment for TSP.

Fair comparison with NS-SE using same token budget.

Usage:
    # Single instance
    uv run python scripts/exp_llamea_tsp.py \
        --instance data/instances/tsp/berlin52.tsp \
        --max-tokens 500000 \
        --model gpt-4o-mini

    # Multiple instances (explicit list)
    uv run python scripts/exp_llamea_tsp.py \
        --instance data/instances/tsp/berlin52.tsp data/instances/tsp/pr76.tsp \
        --max-tokens 500000 \
        --model gpt-4o-mini

    # Multiple instances (from directory)
    uv run python scripts/exp_llamea_tsp.py \
        --instances-dir data/instances/tsp \
        --n-instances 3 \
        --max-tokens 500000 \
        --model gpt-4o-mini
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

from src.baselines.llamea_wrapper import (
    LLaMEABaseline,
    create_tsp_fitness_wrapper,
    create_tsp_task_prompt,
)
from src.domains.tsp import TSPDomain, TSPInstance


def load_tsp_instance(path: str) -> tuple[TSPInstance, list[list[float]]]:
    """Load TSP instance and compute distance matrix.

    Args:
        path: Path to .tsp file

    Returns:
        Tuple of (instance, distance_matrix)
    """
    domain = TSPDomain()
    instance = domain.load_instance(Path(path))

    return instance, instance.distance_matrix


def run_experiment(
    instance_path: str,
    model: str,
    max_tokens: int,
    eval_timeout: int = 60,
) -> dict:
    """Run LLaMEA on a single TSP instance.

    Args:
        instance_path: Path to .tsp file
        model: OpenAI model name
        max_tokens: Token budget
        eval_timeout: Timeout per evaluation in seconds

    Returns:
        Results dictionary
    """
    # Load instance
    instance, distance_matrix = load_tsp_instance(instance_path)
    instance_name = Path(instance_path).stem

    print(f"\n{'='*60}")
    print(f"Instance: {instance_name} ({instance.dimension} cities)")
    print(f"Optimal: {instance.optimal_cost or 'unknown'}")
    print(f"Model: {model}")
    print(f"Max tokens: {max_tokens:,}")
    print(f"{'='*60}\n")

    # Create fitness function and prompt
    fitness_fn = create_tsp_fitness_wrapper(distance_matrix, timeout_seconds=eval_timeout)
    task_prompt = create_tsp_task_prompt(instance_name, instance.dimension)

    # Run LLaMEA
    llamea = LLaMEABaseline(
        model=model,
        max_tokens=max_tokens,
        llm_backend="openai",
        eval_timeout=eval_timeout,
    )

    start_time = time.time()
    result = llamea.run(fitness_fn, task_prompt)
    elapsed = time.time() - start_time

    # Calculate gap if optimal known
    gap = None
    if instance.optimal_cost and result.best_fitness < float("inf"):
        gap = 100.0 * (result.best_fitness - instance.optimal_cost) / instance.optimal_cost

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {instance_name}")
    print(f"{'='*60}")
    print(f"Best fitness: {result.best_fitness:.2f}")
    if gap is not None:
        print(f"Gap: {gap:.2f}%")
    print(f"Evaluations: {result.evaluations}")
    print(f"LLM calls: {result.llm_calls}")
    print(f"Tokens: {result.total_tokens:,} (in={result.prompt_tokens:,} out={result.completion_tokens:,})")
    print(f"Cost: ${result.estimated_cost_usd:.4f}")
    print(f"Time: {elapsed:.1f}s (LLM: {result.llm_time_seconds:.1f}s)")
    print(f"Success rate: {result.llm_success_rate:.1f}%")
    if result.stopped_by_token_limit:
        print(f"Stopped by token limit: YES")
    print(f"{'='*60}\n")

    return {
        "instance": instance_name,
        "dimension": instance.dimension,
        "optimal": instance.optimal_cost,
        "best_fitness": result.best_fitness,
        "gap": gap,
        "evaluations": result.evaluations,
        "llm_calls": result.llm_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.estimated_cost_usd,
        "wall_time": elapsed,
        "llm_time": result.llm_time_seconds,
        "success_rate": result.llm_success_rate,
        "stopped_by_token_limit": result.stopped_by_token_limit,
        "best_code": result.best_code,
        "fitness_history": result.fitness_history,
    }


def main():
    parser = argparse.ArgumentParser(description="LLaMEA baseline for TSP")

    # Instance selection
    parser.add_argument(
        "--instance",
        type=str,
        nargs="+",
        help="Path(s) to .tsp instance(s)",
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        help="Directory with .tsp instances",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=3,
        help="Number of instances to use (sorted by size)",
    )

    # Model and budget
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model (default: gpt-5.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500000,
        help="Token budget (default: 500000)",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=60,
        help="Timeout per evaluation in seconds (default: 60)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/llamea/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable required")
        sys.exit(1)

    # Collect instances
    instances = []
    domain = TSPDomain()
    if args.instance:
        instances = args.instance  # Already a list due to nargs='+'
    elif args.instances_dir:
        tsp_files = sorted(Path(args.instances_dir).glob("*.tsp"))
        # Sort by size (dimension)
        sized = []
        for f in tsp_files:
            try:
                inst = domain.load_instance(f)
                sized.append((inst.dimension, str(f)))
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
                continue
        sized.sort()
        instances = [f for _, f in sized[:args.n_instances]]
    else:
        print("ERROR: Must specify --instance or --instances-dir")
        sys.exit(1)

    print(f"\nLLaMEA TSP Experiment")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Instances: {len(instances)}")
    for inst in instances:
        print(f"  - {Path(inst).stem}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.model.replace('.', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    total_tokens = 0
    total_cost = 0.0

    for instance_path in instances:
        # Calculate remaining tokens
        remaining_tokens = args.max_tokens - total_tokens
        if remaining_tokens <= 0:
            print(f"\nToken budget exhausted. Stopping.")
            break

        result = run_experiment(
            instance_path=instance_path,
            model=args.model,
            max_tokens=remaining_tokens,
            eval_timeout=args.eval_timeout,
        )

        all_results.append(result)
        total_tokens += result["total_tokens"]
        total_cost += result["cost_usd"]

        # Save best code
        code_path = output_dir / f"best_program_{result['instance']}.py"
        with open(code_path, "w") as f:
            f.write(f'"""Best LLaMEA solution for {result["instance"]}.\n\n')
            f.write(f'Fitness: {result["best_fitness"]:.2f}\n')
            if result["gap"] is not None:
                f.write(f'Gap: {result["gap"]:.2f}%\n')
            f.write(f'"""\n\n')
            f.write(result["best_code"])

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")

    gaps = [r["gap"] for r in all_results if r["gap"] is not None]
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        print(f"Avg Gap: {avg_gap:.2f}%")

    print(f"Total Tokens: {total_tokens:,}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Results saved to: {output_dir}")

    # Per-instance gaps
    print(f"\n--- Per-Instance Gaps ---")
    for r in all_results:
        gap_str = f"{r['gap']:.2f}%" if r["gap"] is not None else "N/A"
        print(f"  {r['instance']}: {gap_str}")

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "max_tokens": args.max_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_gap": avg_gap if gaps else None,
            "instances": all_results,
        }, f, indent=2, default=str)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
