#!/usr/bin/env python3
"""Run JSSP with Transfer Learning from TSP.

Uses symbolic rules extracted from a TSP AKG snapshot to guide
optimization on JSSP instances.

This script contains ONLY JSSP-specific logic:
- JSSP baselines (SPT, LPT, random dispatch)
- JSSP evaluation and copy functions
- JSSP instance loading

All transfer logic is in src/akg/transfer/.

Usage:
    uv run python scripts/run_jssp_transfer.py \
        --snapshot experiments/nsgge/results/.../akg_snapshot.json \
        --instance data/instances/jssp/ft06.txt \
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
from src.geakg.transfer.adapters.jssp_adapter import JSSPSchedule
from src.domains.jssp import JSSPDomain, JSSPSolution, create_sample_jssp_instance


# =============================================================================
# ILS BASELINE
# =============================================================================

def ils_jssp(instance, domain: JSSPDomain, time_limit: float, seed: int = 42) -> tuple:
    """Iterated Local Search for JSSP.

    Uses adjacent swap local search with random perturbation.

    Args:
        instance: JSSP instance
        domain: JSSP domain
        time_limit: Time limit in seconds
        seed: Random seed

    Returns:
        (best_solution, best_makespan, iterations)
    """
    random.seed(seed)
    n = len(instance.jobs[0])  # operations per job

    def evaluate(schedule: list) -> int:
        """Evaluate schedule makespan."""
        sol = JSSPSolution(schedule=schedule)
        return domain.evaluate_solution(sol, instance)

    def local_search(schedule: list) -> tuple:
        """First-improvement adjacent-swap local search."""
        current = list(schedule)
        current_makespan = evaluate(current)
        improved = True

        while improved:
            improved = False
            for i in range(len(current) - 1):
                # Try adjacent swap
                current[i], current[i + 1] = current[i + 1], current[i]
                new_makespan = evaluate(current)

                if new_makespan < current_makespan:
                    current_makespan = new_makespan
                    improved = True
                    break
                else:
                    # Undo swap
                    current[i], current[i + 1] = current[i + 1], current[i]

        return current, current_makespan

    def perturb(schedule: list, strength: int = 4) -> list:
        """Random swap perturbation."""
        result = list(schedule)
        for _ in range(strength):
            i, j = random.sample(range(len(result)), 2)
            result[i], result[j] = result[j], result[i]
        return result

    # Initialize with SPT dispatch
    spt_sol = domain.priority_dispatch_solution(instance, priority="spt")
    current = list(spt_sol.schedule)
    current, current_makespan = local_search(current)

    best = list(current)
    best_makespan = current_makespan

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(current)

        # Local search
        new_sol, new_makespan = local_search(perturbed)

        # Accept if better (strict ILS)
        if new_makespan < current_makespan:
            current = new_sol
            current_makespan = new_makespan

            if new_makespan < best_makespan:
                best = list(new_sol)
                best_makespan = new_makespan

        iterations += 1

    return best, best_makespan, iterations


# =============================================================================
# JSSP-SPECIFIC BASELINES
# =============================================================================

def spt_dispatch(instance) -> JSSPSolution:
    """Shortest Processing Time dispatch rule."""
    domain = JSSPDomain()
    return domain.priority_dispatch_solution(instance, priority="spt")


def lpt_dispatch(instance) -> JSSPSolution:
    """Longest Processing Time dispatch rule."""
    domain = JSSPDomain()
    return domain.priority_dispatch_solution(instance, priority="lpt")


def random_dispatch(instance) -> JSSPSolution:
    """Random dispatch rule."""
    domain = JSSPDomain()
    return domain.random_solution(instance)


# =============================================================================
# JSSP-SPECIFIC FUNCTIONS FOR SYMBOLIC EXECUTOR
# =============================================================================

def create_jssp_evaluate_fn(domain: JSSPDomain):
    """Create JSSP evaluation function for SymbolicExecutor."""
    def jssp_evaluate(solution, instance):
        if isinstance(solution, JSSPSchedule):
            if solution.makespan > 0:
                return solution.makespan
            return domain.evaluate_solution(
                JSSPSolution(schedule=solution.schedule), instance
            )
        return domain.evaluate_solution(solution, instance)
    return jssp_evaluate


def jssp_copy(solution):
    """Copy JSSP solution."""
    if isinstance(solution, JSSPSchedule):
        return JSSPSchedule(
            schedule=list(solution.schedule),
            makespan=solution.makespan
        )
    return JSSPSolution(schedule=list(solution.schedule))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="JSSP with Transfer Learning from TSP (Symbolic Rules)"
    )
    parser.add_argument("--snapshot", type=str, help="Path to TSP AKG snapshot")
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="JSSP instance file (uses sample if not provided)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="Number of jobs for sample instance"
    )
    parser.add_argument(
        "--n-machines",
        type=int,
        default=6,
        help="Number of machines for sample instance"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="Time limit in seconds"
    )
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    print("=" * 60)
    print("JSSP TRANSFER LEARNING FROM TSP (SYMBOLIC RULES)")
    print("=" * 60)

    # Find snapshot
    snapshot_path = args.snapshot
    if not snapshot_path:
        snapshot_path = find_latest_snapshot_with_operators()
        if not snapshot_path:
            print("\nNo snapshot with synthesized operators found.")
            print("Run TSP training first.")
            return

    print(f"\nSnapshot: {snapshot_path}")

    # Load snapshot
    with open(snapshot_path) as f:
        snapshot = json.load(f)

    synth_ops = snapshot.get("operators", {}).get("synthesized_synth", [])
    print(f"synthesized operators: {len(synth_ops)}")

    if not synth_ops:
        print("ERROR: Snapshot has no synthesized operators!")
        return

    for op_data in synth_ops[:5]:
        op_id = op_data.get("operator_id", "?")
        role = op_data.get("role", "?")
        print(f"  - {op_id} ({role})")
    if len(synth_ops) > 5:
        print(f"  ... and {len(synth_ops) - 5} more")

    # Load JSSP instance
    print(f"\n--- Loading JSSP Instance ---")
    domain = JSSPDomain()

    if args.instance:
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"Instance not found: {instance_path}")
            return
        jssp_instance = domain.load_instance(instance_path)
    else:
        print(f"Using sample instance: {args.n_jobs} jobs x {args.n_machines} machines")
        jssp_instance = create_sample_jssp_instance(
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            seed=42
        )

    print(f"Instance: {jssp_instance.name}")
    print(f"Jobs: {jssp_instance.n_jobs}")
    print(f"Machines: {jssp_instance.n_machines}")
    print(f"Operations: {jssp_instance.dimension}")
    if jssp_instance.optimal_makespan:
        print(f"Optimal: {jssp_instance.optimal_makespan}")

    # Transfer operators
    print(f"\n--- Transfer Learning: TSP -> JSSP ---")
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="jssp",
        target_instance=jssp_instance,
    )

    print(f"Transferred: {result.operators_transferred} operators")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")

    if not result.adapted_operators:
        print("ERROR: No operators were transferred!")
        return

    adapted_operators = result.adapted_operators

    # Baselines
    print(f"\n--- Baselines ---")
    spt_sol = spt_dispatch(jssp_instance)
    spt_makespan = domain.evaluate_solution(spt_sol, jssp_instance)
    print(f"SPT (Shortest Processing Time): {spt_makespan:.0f}")

    lpt_sol = lpt_dispatch(jssp_instance)
    lpt_makespan = domain.evaluate_solution(lpt_sol, jssp_instance)
    print(f"LPT (Longest Processing Time):  {lpt_makespan:.0f}")

    # Use best baseline as starting point
    if spt_makespan <= lpt_makespan:
        best_baseline_sol = spt_sol
        best_baseline_makespan = spt_makespan
        best_baseline_name = "SPT"
    else:
        best_baseline_sol = lpt_sol
        best_baseline_makespan = lpt_makespan
        best_baseline_name = "LPT"

    print(f"\nBest baseline: {best_baseline_name} ({best_baseline_makespan:.0f})")

    # Calculate gap if optimal is known
    optimal = jssp_instance.optimal_makespan
    if optimal:
        spt_gap = 100 * (spt_makespan - optimal) / optimal
        lpt_gap = 100 * (lpt_makespan - optimal) / optimal
        best_gap = 100 * (best_baseline_makespan - optimal) / optimal
        print(f"SPT gap: {spt_gap:.1f}%")
        print(f"LPT gap: {lpt_gap:.1f}%")

    # ILS baseline (same time budget as transfer)
    print(f"\nRunning ILS ({args.time_limit}s)...", end=" ", flush=True)
    ils_schedule, ils_makespan, ils_iters = ils_jssp(
        jssp_instance, domain, args.time_limit, seed=42
    )
    print(f"done ({ils_iters} iterations)")
    print(f"ILS: {ils_makespan}")

    # Extract symbolic rules from snapshot
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

    # Create executor with JSSP-specific functions
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_jssp_evaluate_fn(domain),
        copy_fn=jssp_copy,
        operator_pheromones=operator_pheromones,
        success_frequency=success_frequency,
        global_mode=True,
        verbose=not args.quiet,
    )

    # Convert initial solution
    initial = JSSPSchedule(
        schedule=list(best_baseline_sol.schedule),
        makespan=int(best_baseline_makespan),
    )

    # Execute
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=best_baseline_makespan,
        time_limit=args.time_limit,
        instance=jssp_instance,
    )

    best_sol = exec_result.best_solution
    history = exec_result.history

    final_makespan = best_sol.makespan

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if optimal:
        transfer_gap = 100 * (final_makespan - optimal) / optimal
        baseline_gap = 100 * (best_baseline_makespan - optimal) / optimal
        ils_gap = 100 * (ils_makespan - optimal) / optimal
        print(f"Transfer Learning: {final_makespan:.0f} (gap: {transfer_gap:.1f}%)")
        print(f"ILS:               {ils_makespan:.0f} (gap: {ils_gap:.1f}%)")
        print(f"{best_baseline_name} Baseline:    {best_baseline_makespan:.0f} (gap: {baseline_gap:.1f}%)")
    else:
        print(f"Transfer Learning: {final_makespan:.0f}")
        print(f"ILS:               {ils_makespan:.0f}")
        print(f"{best_baseline_name} Baseline:    {best_baseline_makespan:.0f}")

    # Compare with ILS
    ils_improvement = ils_makespan - final_makespan
    if ils_improvement > 0:
        pct = 100 * ils_improvement / ils_makespan
        print(f"\nTransfer beats ILS by: {ils_improvement:.0f} ({pct:.1f}%)")
    elif ils_improvement < 0:
        pct = 100 * (-ils_improvement) / final_makespan
        print(f"\nILS beats Transfer by: {-ils_improvement:.0f} ({pct:.1f}%)")
    else:
        print(f"\nTransfer and ILS tied")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "instance": jssp_instance.name,
            "n_jobs": jssp_instance.n_jobs,
            "n_machines": jssp_instance.n_machines,
            "optimal": jssp_instance.optimal_makespan,
            "transfer_makespan": final_makespan,
            "transfer_gap": transfer_gap if optimal else None,
            "ils_makespan": ils_makespan,
            "ils_iterations": ils_iters,
            "spt_makespan": spt_makespan,
            "lpt_makespan": lpt_makespan,
            "baseline_makespan": best_baseline_makespan,
            "baseline_name": best_baseline_name,
            "operators_transferred": result.operators_transferred,
            "time_limit": args.time_limit,
            "history": history,
            "best_schedule": best_sol.schedule,
        }

        with open(output_dir / "jssp_transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
