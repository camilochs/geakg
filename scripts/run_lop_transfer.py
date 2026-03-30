#!/usr/bin/env python3
"""Run LOP with Transfer Learning from TSP.

Uses symbolic rules extracted from a TSP AKG snapshot to guide
optimization on LOP instances.

Usage:
    uv run python scripts/run_lop_transfer.py \
        --instance data/instances/lop/be75eec.txt \
        --time-limit 60
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.geakg.transfer import (
    TransferManager,
    extract_symbolic_rules,
    SymbolicExecutor,
)
from src.geakg.transfer.snapshot_utils import find_latest_snapshot_with_operators
from src.geakg.transfer.adapters.lop_adapter import LOPPermutation
from src.domains.lop import LOPDomain, LOPSolution, create_sample_lop_instance


# =============================================================================
# ILS BASELINE
# =============================================================================

def ils_lop(instance, domain: LOPDomain, time_limit: float, seed: int = 42) -> tuple:
    """Iterated Local Search for LOP.

    Uses swap-based local search with random perturbation.
    Note: LOP is a MAXIMIZATION problem.

    Args:
        instance: LOP instance
        domain: LOP domain
        time_limit: Time limit in seconds
        seed: Random seed

    Returns:
        (best_permutation, best_value, iterations)
    """
    random.seed(seed)
    n = instance.n

    def evaluate(permutation: list) -> int:
        """Evaluate LOP value (sum of elements above diagonal)."""
        value = 0
        for i in range(n):
            for j in range(i + 1, n):
                value += instance.matrix[permutation[i]][permutation[j]]
        return value

    def local_search(permutation: list) -> tuple:
        """First-improvement swap-based local search."""
        current = list(permutation)
        current_value = evaluate(current)
        improved = True

        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Try swap
                    current[i], current[j] = current[j], current[i]
                    new_value = evaluate(current)

                    if new_value > current_value:  # MAXIMIZE
                        current_value = new_value
                        improved = True
                        break
                    else:
                        # Undo swap
                        current[i], current[j] = current[j], current[i]

                if improved:
                    break

        return current, current_value

    def perturb(permutation: list, strength: int = 3) -> list:
        """Random swap perturbation."""
        result = list(permutation)
        for _ in range(strength):
            i, j = random.sample(range(n), 2)
            result[i], result[j] = result[j], result[i]
        return result

    # Initialize with Becker heuristic
    becker_sol = domain.becker_solution(instance)
    current = list(becker_sol.permutation)
    current, current_value = local_search(current)

    best = list(current)
    best_value = current_value

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(current)

        # Local search
        new_perm, new_value = local_search(perturbed)

        # Accept if better (strict ILS - maximize)
        if new_value > current_value:
            current = new_perm
            current_value = new_value

            if new_value > best_value:
                best = list(new_perm)
                best_value = new_value

        iterations += 1

    return best, best_value, iterations


# =============================================================================
# LOP-SPECIFIC FUNCTIONS FOR SYMBOLIC EXECUTOR
# =============================================================================

def create_lop_evaluate_fn(domain: LOPDomain):
    """Create LOP evaluation function for SymbolicExecutor."""
    def lop_evaluate(solution, instance):
        if isinstance(solution, LOPPermutation):
            # Return negative value (we minimize, LOP maximizes)
            return -solution.value if solution.value != 0 else domain.evaluate_solution(
                LOPSolution(permutation=solution.permutation), instance
            )
        return domain.evaluate_solution(solution, instance)
    return lop_evaluate


def lop_copy(solution):
    """Copy LOP solution."""
    if isinstance(solution, LOPPermutation):
        return LOPPermutation(
            permutation=list(solution.permutation),
            value=solution.value
        )
    return LOPSolution(permutation=list(solution.permutation))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LOP with Transfer Learning from TSP (Symbolic Rules)"
    )
    parser.add_argument("--snapshot", type=str, help="Path to TSP AKG snapshot")
    parser.add_argument("--instance", type=str, help="LOP instance file")
    parser.add_argument("--n", type=int, default=10, help="Size for sample instance")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    print("=" * 60)
    print("LOP TRANSFER LEARNING FROM TSP (SYMBOLIC RULES)")
    print("=" * 60)

    # Find snapshot
    snapshot_path = args.snapshot
    if not snapshot_path:
        snapshot_path = find_latest_snapshot_with_operators()
        if not snapshot_path:
            print("\nNo snapshot with synthesized operators found.")
            return

    print(f"\nSnapshot: {snapshot_path}")

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    synth_ops = snapshot.get("operators", {}).get("synthesized_synth", [])
    print(f"synthesized operators: {len(synth_ops)}")

    if not synth_ops:
        print("ERROR: Snapshot has no synthesized operators!")
        return

    # Load LOP instance
    print(f"\n--- Loading LOP Instance ---")
    domain = LOPDomain()

    if args.instance:
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"Instance not found: {instance_path}")
            return
        lop_instance = domain.load_instance(instance_path)
    else:
        print(f"Using sample instance: n={args.n}")
        lop_instance = create_sample_lop_instance(n=args.n, seed=42)

    print(f"Instance: {lop_instance.name}")
    print(f"Size: {lop_instance.n}")
    if lop_instance.optimal_value:
        print(f"Optimal: {lop_instance.optimal_value}")

    # Transfer operators
    print(f"\n--- Transfer Learning: TSP -> LOP ---")
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="lop",
        target_instance=lop_instance,
    )

    print(f"Transferred: {result.operators_transferred} operators")

    if not result.adapted_operators:
        print("ERROR: No operators were transferred!")
        return

    adapted_operators = result.adapted_operators

    # Baselines
    print(f"\n--- Baselines ---")

    # Becker's heuristic (1967) - domain-specific for LOP
    becker_sol = domain.becker_solution(lop_instance)
    becker_value = becker_sol.value
    print(f"Becker (1967): {becker_value}")

    greedy_sol = domain.greedy_solution(lop_instance)
    greedy_value = greedy_sol.value
    print(f"Greedy insertion: {greedy_value}")

    random_sol = domain.random_solution(lop_instance)
    random_value = random_sol.value
    print(f"Random: {random_value}")

    # Use Becker as starting point (domain-specific heuristic)
    best_baseline_sol = becker_sol
    best_baseline_value = becker_value
    best_baseline_name = "Becker"

    print(f"\nBaseline: {best_baseline_name} ({best_baseline_value})")

    # ILS baseline (same time budget as transfer)
    print(f"\nRunning ILS ({args.time_limit}s)...", end=" ", flush=True)
    ils_perm, ils_value, ils_iters = ils_lop(
        lop_instance, domain, args.time_limit, seed=42
    )
    print(f"done ({ils_iters} iterations)")
    print(f"ILS: {ils_value}")

    # Extract symbolic rules
    print(f"\n--- Transfer Learning Optimization ---")
    rule_engine = extract_symbolic_rules(snapshot)

    # Extract operator pheromones from snapshot
    operator_pheromones = {}
    if "pheromones" in snapshot:
        operator_pheromones = snapshot["pheromones"].get("operator_level", {})

    # Extract success frequency from successful paths
    success_frequency = {}
    for path in snapshot.get("successful_paths", []):
        for op_id in path.get("operators", []):
            success_frequency[op_id] = success_frequency.get(op_id, 0) + 1

    print(f"Using symbolic rules (pure transfer):")
    print(f"  Stagnation threshold: {rule_engine.stagnation_threshold}")
    print(f"  Operator pheromones: {len(operator_pheromones)}")
    print(f"  Success frequency entries: {len(success_frequency)}")

    # Create executor
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_lop_evaluate_fn(domain),
        copy_fn=lop_copy,
        operator_pheromones=operator_pheromones,
        success_frequency=success_frequency,
        global_mode=True,
        verbose=not args.quiet,
    )

    # Convert initial solution (negative value for cost)
    initial = LOPPermutation(
        permutation=list(best_baseline_sol.permutation),
        value=best_baseline_value,
    )

    # Execute
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=-best_baseline_value,  # Negative because we minimize
        time_limit=args.time_limit,
        instance=lop_instance,
    )

    best_sol = exec_result.best_solution
    # Recalculate value (cost is negative of value)
    final_value = -int(exec_result.best_cost)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    optimal = lop_instance.optimal_value
    if optimal:
        transfer_gap = 100 * (optimal - final_value) / optimal
        baseline_gap = 100 * (optimal - best_baseline_value) / optimal
        ils_gap = 100 * (optimal - ils_value) / optimal
        print(f"Transfer Learning: {final_value} (gap: {transfer_gap:.1f}%)")
        print(f"ILS:               {ils_value} (gap: {ils_gap:.1f}%)")
        print(f"{best_baseline_name} Baseline: {best_baseline_value} (gap: {baseline_gap:.1f}%)")
    else:
        print(f"Transfer Learning: {final_value}")
        print(f"ILS:               {ils_value}")
        print(f"{best_baseline_name} Baseline: {best_baseline_value}")

    # Compare with ILS (LOP is maximization - higher is better)
    ils_improvement = final_value - ils_value
    if ils_improvement > 0:
        pct = 100 * ils_improvement / ils_value
        print(f"\nTransfer beats ILS by: {ils_improvement} ({pct:.1f}%)")
    elif ils_improvement < 0:
        pct = 100 * (-ils_improvement) / final_value
        print(f"\nILS beats Transfer by: {-ils_improvement} ({pct:.1f}%)")
    else:
        print(f"\nTransfer and ILS tied")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "instance": lop_instance.name,
            "n": lop_instance.n,
            "optimal": optimal,
            "transfer_value": final_value,
            "ils_value": ils_value,
            "ils_iterations": ils_iters,
            "becker_value": becker_value,
            "greedy_value": greedy_value,
            "random_value": random_value,
            "operators_transferred": result.operators_transferred,
            "time_limit": args.time_limit,
        }

        with open(output_dir / "lop_transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
