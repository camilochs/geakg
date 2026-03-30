#!/usr/bin/env python3
"""Run LLaMEA on multiple TSP instances with token budget limit.

Usage:
    uv run python scripts/exp_llamea_multi_instance.py \
        --instances data/instances/tsp_generated/tsp50_001.tsp,data/instances/tsp_generated/tsp100_001.tsp \
        --model gpt-4o \
        --max-tokens 200000
"""

import argparse
import os
import time
from pathlib import Path

from src.domains.tsp import TSPDomain


def load_instances(instance_paths: list[str]) -> list[dict]:
    """Load TSP instances from paths.

    Returns list of dicts with: name, dimension, distance_matrix, optimal
    """
    domain = TSPDomain()
    instances = []

    for path_str in instance_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        instance = domain.load_instance(path)
        instances.append({
            "name": instance.name,
            "dimension": instance.dimension,
            "distance_matrix": instance.distance_matrix,
            "optimal": instance.optimal_cost,
        })
        print(f"Loaded: {instance.name} (n={instance.dimension}, optimal={instance.optimal_cost})")

    return instances


def create_multi_instance_fitness(instances: list[dict], timeout: int = 30):
    """Create fitness function that evaluates on all instances.

    Returns average gap across all instances (lower is better).
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")

    def evaluate_code(code: str) -> float:
        """Evaluate TSP solver code on all instances, return average gap."""
        import random
        import math
        import itertools
        import functools
        import collections

        # Try to compile and extract function
        try:
            exec_globals = {
                "__builtins__": __builtins__,
                "random": random,
                "math": math,
                "itertools": itertools,
                "functools": functools,
                "collections": collections,
            }
            local_vars = {}
            exec(code, exec_globals, local_vars)

            if "solve_tsp" not in local_vars:
                return float("inf")

            solve_fn = local_vars["solve_tsp"]
        except Exception:
            return float("inf")

        # Evaluate on each instance
        gaps = []
        for inst in instances:
            try:
                # Run with timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

                try:
                    tour = solve_fn(inst["distance_matrix"])
                finally:
                    signal.alarm(0)

                # Validate tour
                n = inst["dimension"]
                if not isinstance(tour, list) or len(tour) != n or set(tour) != set(range(n)):
                    gaps.append(float("inf"))
                    continue

                # Calculate tour cost
                dist = inst["distance_matrix"]
                cost = sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

                # Calculate gap
                if inst["optimal"] and inst["optimal"] > 0:
                    gap = 100.0 * (cost - inst["optimal"]) / inst["optimal"]
                else:
                    gap = cost  # No optimal known

                gaps.append(gap)

            except Exception:
                gaps.append(float("inf"))

        # Return average gap (or inf if all failed)
        valid_gaps = [g for g in gaps if g != float("inf")]
        if not valid_gaps:
            return float("inf")

        return sum(valid_gaps) / len(valid_gaps)

    return evaluate_code


def create_tsp_task_prompt(instances: list[dict]) -> str:
    """Create task prompt describing multi-instance TSP task."""
    instance_desc = "\n".join([
        f"  - {inst['name']}: {inst['dimension']} cities, optimal={inst['optimal']}"
        for inst in instances
    ])

    return f"""You are designing a heuristic algorithm for the Traveling Salesman Problem (TSP).

The algorithm will be evaluated on multiple instances:
{instance_desc}

Your task is to write a Python function that takes a distance matrix and returns a tour.
The function should be named `solve_tsp` and have this signature:

```python
def solve_tsp(distance_matrix: list[list[float]]) -> list[int]:
    '''
    Solve TSP instance.

    Args:
        distance_matrix: NxN matrix of distances between cities

    Returns:
        Tour as list of city indices (0 to N-1), visiting each city exactly once
    '''
    # Your implementation here
    pass
```

Requirements:
- Return a valid tour visiting all cities exactly once
- Minimize total tour length (sum of distances along the tour)
- The algorithm should work well across different instance sizes (50-150 cities)
- You can use standard Python libraries (random, math, itertools, collections)
- Be efficient - the algorithm will be evaluated multiple times

Design a clever heuristic that balances solution quality with computation time.
Write only the function implementation."""


def run_llamea_experiment(
    instances: list[dict],
    model: str = "gpt-4o",
    backend: str = "openai",
    max_tokens: int = 200_000,
    n_parents: int = 5,
    n_offspring: int = 5,
    budget: int = 100,
    eval_timeout: int = 30,
    output_file: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Run LLaMEA experiment on multiple instances."""
    import json
    from src.baselines.llamea_wrapper import LLaMEABaseline

    print(f"\n{'='*70}")
    print(f"LLaMEA Experiment")
    print(f"{'='*70}")
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"Max tokens: {max_tokens:,}")
    print(f"Instances: {len(instances)}")
    print(f"Population: {n_parents} parents, {n_offspring} offspring")
    print(f"Budget: {budget} generations")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"{'='*70}\n")

    # Create fitness function
    fitness_fn = create_multi_instance_fitness(instances, timeout=eval_timeout)

    # Create task prompt
    task_prompt = create_tsp_task_prompt(instances)

    # Create and run LLaMEA
    llamea = LLaMEABaseline(
        model=model,
        budget=budget,
        max_tokens=max_tokens,
        n_parents=n_parents,
        n_offspring=n_offspring,
        eval_timeout=eval_timeout,
        llm_backend=backend,
        openai_api_key=os.environ.get("OPENAI_API_KEY") if backend == "openai" else None,
        reasoning_effort=reasoning_effort,
    )

    start_time = time.time()
    result = llamea.run(fitness_fn, task_prompt)
    total_time = time.time() - start_time

    # Print results
    print(f"\n{'='*70}")
    print(f"Results")
    print(f"{'='*70}")
    print(f"Best fitness (avg gap): {result.best_fitness:.2f}%")
    print(f"Total evaluations: {result.evaluations}")
    print(f"LLM calls: {result.llm_calls}")
    print(f"Token usage:")
    print(f"  - Prompt tokens: {result.prompt_tokens:,}")
    print(f"  - Completion tokens: {result.completion_tokens:,}")
    print(f"  - Total tokens: {result.total_tokens:,}")
    print(f"  - Estimated cost: ${result.estimated_cost_usd:.4f}")
    print(f"  - Stopped by token limit: {result.stopped_by_token_limit}")
    print(f"Time:")
    print(f"  - Total wall time: {total_time:.1f}s")
    print(f"  - LLM time: {result.llm_time_seconds:.1f}s")
    print(f"  - Compute time: {result.compute_time_seconds:.1f}s")
    print(f"  - LLM success rate: {result.llm_success_rate:.1f}%")

    # Evaluate best solution on each instance
    per_instance_results = []
    if result.best_code:
        print(f"\n--- Per-Instance Results ---")

        import random, math, itertools, functools, collections
        exec_globals = {
            "__builtins__": __builtins__,
            "random": random, "math": math, "itertools": itertools,
            "functools": functools, "collections": collections,
        }
        local_vars = {}
        try:
            exec(result.best_code, exec_globals, local_vars)
            solve_fn = local_vars.get("solve_tsp")

            if solve_fn:
                for inst in instances:
                    inst_result = {
                        "name": inst["name"],
                        "dimension": inst["dimension"],
                        "optimal": inst["optimal"],
                    }
                    try:
                        tour = solve_fn(inst["distance_matrix"])
                        n = inst["dimension"]
                        if isinstance(tour, list) and len(tour) == n and set(tour) == set(range(n)):
                            dist = inst["distance_matrix"]
                            cost = sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))
                            inst_result["cost"] = cost
                            if inst["optimal"] and inst["optimal"] > 0:
                                gap = 100.0 * (cost - inst["optimal"]) / inst["optimal"]
                                inst_result["gap"] = gap
                                print(f"  {inst['name']}: cost={cost:.0f}, gap={gap:.2f}%")
                            else:
                                inst_result["gap"] = None
                                print(f"  {inst['name']}: cost={cost:.0f}")
                        else:
                            inst_result["cost"] = None
                            inst_result["gap"] = None
                            inst_result["error"] = "INVALID TOUR"
                            print(f"  {inst['name']}: INVALID TOUR")
                    except Exception as e:
                        inst_result["cost"] = None
                        inst_result["gap"] = None
                        inst_result["error"] = str(e)
                        print(f"  {inst['name']}: ERROR - {e}")
                    per_instance_results.append(inst_result)
        except Exception as e:
            print(f"  Could not evaluate best solution: {e}")

        print(f"\n--- Best Code ---")
        print(result.best_code[:500] + "..." if len(result.best_code) > 500 else result.best_code)

    print(f"{'='*70}\n")

    # Build final results
    final_results = {
        "model": model,
        "max_tokens": max_tokens,
        "budget": budget,
        "n_parents": n_parents,
        "n_offspring": n_offspring,
        "best_fitness": result.best_fitness,
        "best_code": result.best_code,
        "total_tokens": result.total_tokens,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.estimated_cost_usd,
        "wall_time_seconds": total_time,
        "llm_time_seconds": result.llm_time_seconds,
        "compute_time_seconds": result.compute_time_seconds,
        "evaluations": result.evaluations,
        "llm_calls": result.llm_calls,
        "stopped_by_token_limit": result.stopped_by_token_limit,
        "per_instance_results": per_instance_results,
        "tokens_per_llm_call": result.total_tokens / result.llm_calls if result.llm_calls > 0 else 0,
    }

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"Results saved to: {output_file}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Run LLaMEA on multiple TSP instances")
    parser.add_argument("--instances", type=str, required=True,
                        help="Comma-separated list of instance paths")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model to use")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "ollama"],
                        help="LLM backend: openai or ollama (default: openai)")
    parser.add_argument("--max-tokens", type=int, default=200_000,
                        help="Maximum token budget")
    parser.add_argument("--budget", type=int, default=100,
                        help="Number of generations")
    parser.add_argument("--n-parents", type=int, default=5,
                        help="Population size (parents)")
    parser.add_argument("--n-offspring", type=int, default=5,
                        help="Offspring per generation")
    parser.add_argument("--eval-timeout", type=int, default=30,
                        help="Timeout per evaluation in seconds")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for GPT-5 models (default: none for gpt-5.1/5.2, medium for gpt-5)")

    args = parser.parse_args()

    # Set API key in environment if provided (only needed for OpenAI)
    if args.backend == "openai":
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OpenAI API key required. Use --api-key or set OPENAI_API_KEY env var")
            return

    # Load instances
    instance_paths = [p.strip() for p in args.instances.split(",")]
    instances = load_instances(instance_paths)

    if not instances:
        print("Error: No valid instances loaded")
        return

    # Run experiment
    run_llamea_experiment(
        instances=instances,
        model=args.model,
        backend=args.backend,
        max_tokens=args.max_tokens,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        budget=args.budget,
        eval_timeout=args.eval_timeout,
        output_file=args.output,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
