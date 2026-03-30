#!/usr/bin/env python3
"""Run PFSP with Transfer Learning from TSP.

Uses symbolic rules extracted from a TSP AKG snapshot to guide
optimization on PFSP instances.

This script contains ONLY PFSP-specific logic:
- PFSP baselines (NEH, SPT)
- PFSP evaluation and copy functions
- PFSP instance loading

All transfer logic is in src/akg/transfer/.

Usage:
    uv run python scripts/run_pfsp_transfer.py \
        --snapshot experiments/nsgge/results/.../akg_snapshot.json \
        --time-limit 60
"""

import argparse
import json
from pathlib import Path

from src.geakg.transfer import (
    TransferManager,
    extract_symbolic_rules,
    SymbolicExecutor,
)
from src.geakg.transfer.snapshot_utils import find_latest_snapshot_with_operators
from src.geakg.transfer.adapters.pfsp_adapter import PFSPSequence
from src.domains.pfsp import PFSPDomain, PFSPSolution, create_sample_pfsp_instance


# =============================================================================
# PFSP-SPECIFIC FUNCTIONS FOR SYMBOLIC EXECUTOR
# =============================================================================

def create_pfsp_evaluate_fn(domain: PFSPDomain):
    """Create PFSP evaluation function for SymbolicExecutor."""
    def pfsp_evaluate(solution, instance):
        if isinstance(solution, PFSPSequence):
            if solution.makespan > 0:
                return solution.makespan
            return domain.evaluate_solution(
                PFSPSolution(sequence=solution.sequence), instance
            )
        return domain.evaluate_solution(solution, instance)
    return pfsp_evaluate


def pfsp_copy(solution):
    """Copy PFSP solution."""
    if isinstance(solution, PFSPSequence):
        return PFSPSequence(
            sequence=list(solution.sequence),
            makespan=solution.makespan
        )
    return PFSPSolution(sequence=list(solution.sequence))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PFSP with Transfer Learning from TSP (Symbolic Rules)"
    )
    parser.add_argument("--snapshot", type=str, help="Path to TSP AKG snapshot")
    parser.add_argument("--instance", type=str, help="PFSP instance file")
    parser.add_argument("--n-jobs", type=int, default=20, help="Jobs for sample instance")
    parser.add_argument("--n-machines", type=int, default=5, help="Machines for sample instance")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit in seconds")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    print("=" * 60)
    print("PFSP TRANSFER LEARNING FROM TSP (SYMBOLIC RULES)")
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

    # Load PFSP instance
    print(f"\n--- Loading PFSP Instance ---")
    domain = PFSPDomain()

    if args.instance:
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"Instance not found: {instance_path}")
            return
        pfsp_instance = domain.load_instance(instance_path)
    else:
        print(f"Using sample instance: {args.n_jobs} jobs x {args.n_machines} machines")
        pfsp_instance = create_sample_pfsp_instance(
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            seed=42
        )

    print(f"Instance: {pfsp_instance.name}")
    print(f"Jobs: {pfsp_instance.n_jobs}")
    print(f"Machines: {pfsp_instance.n_machines}")
    if pfsp_instance.optimal_makespan:
        print(f"Optimal: {pfsp_instance.optimal_makespan}")

    # Transfer operators
    print(f"\n--- Transfer Learning: TSP -> PFSP ---")
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="pfsp",
        target_instance=pfsp_instance,
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
    neh_sol = domain.neh_solution(pfsp_instance)
    neh_makespan = neh_sol.makespan
    print(f"NEH (Nawaz-Enscore-Ham): {neh_makespan}")

    spt_sol = domain.spt_solution(pfsp_instance)
    spt_makespan = spt_sol.makespan
    print(f"SPT (Shortest Proc. Time): {spt_makespan}")

    # Use SPT as starting point (weaker baseline, more room to improve)
    best_baseline_sol = spt_sol
    best_baseline_makespan = spt_makespan
    best_baseline_name = "SPT"

    print(f"\nBest baseline: {best_baseline_name} ({best_baseline_makespan})")

    # Extract symbolic rules
    print(f"\n--- Transfer Learning Optimization ---")
    rule_engine = extract_symbolic_rules(snapshot)

    print(f"Using symbolic rules (pure transfer):")
    print(f"  Stagnation threshold: {rule_engine.stagnation_threshold}")
    print(f"  Climb threshold: {rule_engine.climb_threshold}")
    print(f"  Max failed perturbations: {rule_engine.max_failed_perturbations}")

    # Create executor
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_pfsp_evaluate_fn(domain),
        copy_fn=pfsp_copy,
        verbose=not args.quiet,
    )

    # Convert initial solution
    initial = PFSPSequence(
        sequence=list(best_baseline_sol.sequence),
        makespan=best_baseline_makespan,
    )

    # Execute
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=best_baseline_makespan,
        time_limit=args.time_limit,
        instance=pfsp_instance,
    )

    best_sol = exec_result.best_solution

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    optimal = pfsp_instance.optimal_makespan
    if optimal:
        transfer_gap = 100 * (best_sol.makespan - optimal) / optimal
        baseline_gap = 100 * (best_baseline_makespan - optimal) / optimal
        print(f"Transfer Learning: {best_sol.makespan} (gap: {transfer_gap:.1f}%)")
        print(f"{best_baseline_name} Baseline: {best_baseline_makespan} (gap: {baseline_gap:.1f}%)")
    else:
        print(f"Transfer Learning: {best_sol.makespan}")
        print(f"{best_baseline_name} Baseline: {best_baseline_makespan}")

    improvement = best_baseline_makespan - best_sol.makespan
    if improvement > 0:
        pct = 100 * improvement / best_baseline_makespan
        print(f"\nImprovement over {best_baseline_name}: {improvement} ({pct:.1f}%)")
    else:
        print(f"\nNo improvement over {best_baseline_name} baseline")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "instance": pfsp_instance.name,
            "n_jobs": pfsp_instance.n_jobs,
            "n_machines": pfsp_instance.n_machines,
            "optimal": optimal,
            "transfer_makespan": best_sol.makespan,
            "neh_makespan": neh_makespan,
            "spt_makespan": spt_makespan,
            "operators_transferred": result.operators_transferred,
            "time_limit": args.time_limit,
        }

        with open(output_dir / "pfsp_transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
