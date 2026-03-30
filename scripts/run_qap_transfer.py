#!/usr/bin/env python3
"""Run QAP with Transfer Learning from TSP.

Uses symbolic rules extracted from a TSP AKG snapshot to guide
optimization on QAP instances.

Usage:
    uv run python scripts/run_qap_transfer.py \
        --instance data/instances/qap/nug12.txt \
        --time-limit 60

    # Run all instances:
    uv run python scripts/run_qap_transfer.py --all
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
from src.geakg.transfer.adapters.qap_adapter import QAPAssignment
from src.domains.qap import QAPDomain, QAPSolution, create_sample_qap_instance


# =============================================================================
# QAP INSTANCES CONFIGURATION
# =============================================================================

# Instance directory
QAP_INSTANCE_DIR = Path("experiments/nsse-transfer/instances/qap")

# All available QAP instances with their optimal/best known solutions
QAP_INSTANCES = {
    # Small (n <= 25)
    "nug12.dat": {"n": 12, "optimal": 578},
    "chr12a.dat": {"n": 12, "optimal": 9552},
    "nug15.dat": {"n": 15, "optimal": 1150},
    "chr15a.dat": {"n": 15, "optimal": 9896},
    "nug20.dat": {"n": 20, "optimal": 2570},
    "chr20a.dat": {"n": 20, "optimal": 2192},
    "tai20a.dat": {"n": 20, "optimal": 703482},
    "nug25.dat": {"n": 25, "optimal": 3744},
    "chr25a.dat": {"n": 25, "optimal": 3796},
    # Medium (25 < n <= 50)
    "nug30.dat": {"n": 30, "optimal": 6124},
    "tai50a.dat": {"n": 50, "optimal": 4938796},
    # Large (n > 50)
    "tai80a.dat": {"n": 80, "optimal": 13499184},
    "sko100a.dat": {"n": 100, "optimal": 152002},
    "wil100.dat": {"n": 100, "optimal": 273038},
    "tai100a.dat": {"n": 100, "optimal": 21052466},
    "tai150b.dat": {"n": 150, "optimal": 498896643},
    "tai256c.dat": {"n": 256, "optimal": 44759294},
}


# =============================================================================
# ILS BASELINE
# =============================================================================

def ils_qap(instance, domain: QAPDomain, time_limit: float, seed: int = 42) -> tuple:
    """Iterated Local Search for QAP.

    Uses swap-based local search with random perturbation.

    Args:
        instance: QAP instance
        domain: QAP domain
        time_limit: Time limit in seconds
        seed: Random seed

    Returns:
        (best_solution, best_cost)
    """
    random.seed(seed)
    n = instance.n

    def evaluate(assignment: list) -> int:
        """Evaluate assignment cost."""
        cost = 0
        for i in range(n):
            for j in range(n):
                cost += instance.flow_matrix[i][j] * instance.distance_matrix[assignment[i]][assignment[j]]
        return cost

    def local_search(assignment: list) -> tuple:
        """First-improvement swap-based local search."""
        current = list(assignment)
        current_cost = evaluate(current)
        improved = True

        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Try swap
                    current[i], current[j] = current[j], current[i]
                    new_cost = evaluate(current)

                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        break
                    else:
                        # Undo swap
                        current[i], current[j] = current[j], current[i]

                if improved:
                    break

        return current, current_cost

    def perturb(assignment: list, strength: int = 3) -> list:
        """Random swap perturbation."""
        result = list(assignment)
        for _ in range(strength):
            i, j = random.sample(range(n), 2)
            result[i], result[j] = result[j], result[i]
        return result

    # Initialize with random solution
    current = list(range(n))
    random.shuffle(current)
    current, current_cost = local_search(current)

    best = list(current)
    best_cost = current_cost

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(current)

        # Local search
        new_sol, new_cost = local_search(perturbed)

        # Accept if better (strict ILS)
        if new_cost < current_cost:
            current = new_sol
            current_cost = new_cost

            if new_cost < best_cost:
                best = list(new_sol)
                best_cost = new_cost

        iterations += 1

    return best, best_cost, iterations


# =============================================================================
# QAP-SPECIFIC FUNCTIONS FOR SYMBOLIC EXECUTOR
# =============================================================================

def create_qap_evaluate_fn(domain: QAPDomain):
    """Create QAP evaluation function for SymbolicExecutor."""
    def qap_evaluate(solution, instance):
        if isinstance(solution, QAPAssignment):
            if solution.cost > 0:
                return solution.cost
            return domain.evaluate_solution(
                QAPSolution(assignment=solution.assignment), instance
            )
        return domain.evaluate_solution(solution, instance)
    return qap_evaluate


def qap_copy(solution):
    """Copy QAP solution."""
    if isinstance(solution, QAPAssignment):
        return QAPAssignment(
            assignment=list(solution.assignment),
            cost=solution.cost
        )
    return QAPSolution(assignment=list(solution.assignment))


# =============================================================================
# MAIN
# =============================================================================

def run_single_instance(
    instance_name: str,
    instance_info: dict,
    snapshot_path: str,
    snapshot: dict,
    pool_data: dict,
    time_limit: float,
    quiet: bool = False,
    seed: int = 42,
) -> dict:
    """Run transfer learning on a single QAP instance.

    Returns dict with results.
    """
    domain = QAPDomain()
    instance_path = QAP_INSTANCE_DIR / instance_name

    if not instance_path.exists():
        return {"instance": instance_name, "status": "not_found"}

    qap_instance = domain.load_instance(instance_path)
    optimal = instance_info.get("optimal")

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}, n={qap_instance.n}")
        if optimal:
            print(f"Optimal: {optimal}")

    # Transfer operators
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="qap",
        target_instance=qap_instance,
    )

    if not result.adapted_operators:
        return {"instance": instance_name, "status": "no_operators"}

    adapted_operators = result.adapted_operators

    # Baselines
    gl_sol = domain.gilmore_lawler_solution(qap_instance)
    gl_cost = gl_sol.cost

    # ILS baseline
    if not quiet:
        print(f"Running ILS ({time_limit}s)...", end=" ", flush=True)
    ils_assignment, ils_cost, ils_iters = ils_qap(
        qap_instance, domain, time_limit, seed=seed
    )
    if not quiet:
        print(f"done ({ils_iters} iters)")

    # Extract symbolic rules
    rule_engine = extract_symbolic_rules(snapshot)
    operator_pheromones = {}
    if "pheromones" in snapshot:
        operator_pheromones = snapshot["pheromones"].get("operator_level", {})

    success_frequency = {}
    for path in snapshot.get("successful_paths", []):
        for op_id in path.get("operators", []):
            success_frequency[op_id] = success_frequency.get(op_id, 0) + 1

    # Create executor
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_qap_evaluate_fn(domain),
        copy_fn=qap_copy,
        operator_pheromones=operator_pheromones,
        success_frequency=success_frequency,
        global_mode=True,
        verbose=False,
    )

    # Initial solution from Gilmore-Lawler
    initial = QAPAssignment(
        assignment=list(gl_sol.assignment),
        cost=gl_cost,
    )

    # Execute
    if not quiet:
        print(f"Running NS-SE Transfer ({time_limit}s)...", end=" ", flush=True)
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=gl_cost,
        time_limit=time_limit,
        instance=qap_instance,
    )
    if not quiet:
        print("done")

    final_cost = exec_result.best_solution.cost

    # Calculate gaps
    transfer_gap = None
    ils_gap = None
    if optimal:
        transfer_gap = 100 * (final_cost - optimal) / optimal
        ils_gap = 100 * (ils_cost - optimal) / optimal

    result_dict = {
        "instance": instance_name,
        "n": qap_instance.n,
        "optimal": optimal,
        "transfer_cost": final_cost,
        "transfer_gap": transfer_gap,
        "ils_cost": ils_cost,
        "ils_gap": ils_gap,
        "ils_iterations": ils_iters,
        "gl_cost": gl_cost,
        "operators_transferred": result.operators_transferred,
        "time_limit": time_limit,
        "status": "ok",
    }

    if not quiet:
        print(f"  NS-SE Transfer: {final_cost}" + (f" (gap: {transfer_gap:.1f}%)" if transfer_gap else ""))
        print(f"  ILS:            {ils_cost}" + (f" (gap: {ils_gap:.1f}%)" if ils_gap else ""))
        if final_cost < ils_cost:
            improvement = 100 * (ils_cost - final_cost) / ils_cost
            print(f"  --> NS-SE wins by {improvement:.1f}%")
        elif ils_cost < final_cost:
            improvement = 100 * (final_cost - ils_cost) / final_cost
            print(f"  --> ILS wins by {improvement:.1f}%")

    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="QAP with Transfer Learning from TSP (Symbolic Rules)"
    )
    parser.add_argument("--snapshot", type=str, help="Path to TSP AKG snapshot")
    parser.add_argument("--instance", type=str, help="QAP instance file")
    parser.add_argument("--all", action="store_true", help="Run all QAP instances")
    parser.add_argument("--n", type=int, default=12, help="Size for sample instance")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit (or 'auto' for n-based)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("QAP TRANSFER LEARNING FROM TSP (SYMBOLIC RULES)")
    print("=" * 60)

    # Find snapshot
    snapshot_path = args.snapshot
    if not snapshot_path:
        snapshot_path = find_latest_snapshot_with_operators()
        if not snapshot_path:
            print("\nNo snapshot with synthesized operators found.")
            return

    print(f"\nSnapshot: {snapshot_path}")

    # Check for refined_pool.json (actual operators)
    snapshot_dir = Path(snapshot_path).parent
    pool_path = snapshot_dir / "refined_pool.json"

    if not pool_path.exists():
        print(f"ERROR: No refined_pool.json found in {snapshot_dir}")
        return

    with open(pool_path) as f:
        pool_data = json.load(f)

    operators_by_role = pool_data.get("operators_by_role", {})
    total_ops = sum(len(ops) for ops in operators_by_role.values())
    print(f"Operators in pool: {total_ops}")

    if total_ops == 0:
        print("ERROR: No operators in refined_pool.json!")
        return

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    # =========================================================================
    # RUN ALL INSTANCES MODE
    # =========================================================================
    if args.all:
        print(f"\n--- Running ALL {len(QAP_INSTANCES)} QAP instances ---")
        all_results = []

        # Sort by size (largest first)
        sorted_instances = sorted(
            QAP_INSTANCES.items(),
            key=lambda x: x[1]["n"],
            reverse=True
        )

        for instance_name, instance_info in sorted_instances:
            n = instance_info["n"]
            # Time limit: use n seconds (same as TSP convention)
            time_limit = args.time_limit if args.time_limit != 60.0 else float(n)

            result = run_single_instance(
                instance_name=instance_name,
                instance_info=instance_info,
                snapshot_path=snapshot_path,
                snapshot=snapshot,
                pool_data=pool_data,
                time_limit=time_limit,
                quiet=args.quiet,
                seed=args.seed,
            )
            all_results.append(result)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: QAP Transfer Learning Results")
        print("=" * 70)
        print(f"{'Instance':<15} {'n':>5} {'NS-SE Gap':>12} {'ILS Gap':>12} {'Winner':>10}")
        print("-" * 70)

        nsse_wins = 0
        ils_wins = 0
        for r in sorted(all_results, key=lambda x: x.get("n", 0)):
            if r["status"] != "ok":
                print(f"{r['instance']:<15} {'-':>5} {'ERROR':>12}")
                continue

            tg = r.get("transfer_gap")
            ig = r.get("ils_gap")
            tg_str = f"{tg:.1f}%" if tg is not None else "-"
            ig_str = f"{ig:.1f}%" if ig is not None else "-"

            if r["transfer_cost"] < r["ils_cost"]:
                winner = "NS-SE"
                nsse_wins += 1
            elif r["ils_cost"] < r["transfer_cost"]:
                winner = "ILS"
                ils_wins += 1
            else:
                winner = "TIE"

            print(f"{r['instance']:<15} {r['n']:>5} {tg_str:>12} {ig_str:>12} {winner:>10}")

        print("-" * 70)
        print(f"NS-SE wins: {nsse_wins}, ILS wins: {ils_wins}")

        # Save results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "qap_all_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {output_dir / 'qap_all_results.json'}")

        return

    # =========================================================================
    # SINGLE INSTANCE MODE (original behavior)
    # =========================================================================
    # Load QAP instance
    print(f"\n--- Loading QAP Instance ---")
    domain = QAPDomain()

    if args.instance:
        instance_path = Path(args.instance)
        if not instance_path.exists():
            print(f"Instance not found: {instance_path}")
            return
        qap_instance = domain.load_instance(instance_path)
    else:
        print(f"Using sample instance: n={args.n}")
        qap_instance = create_sample_qap_instance(n=args.n, seed=42)

    print(f"Instance: {qap_instance.name}")
    print(f"Size: {qap_instance.n}")
    if qap_instance.optimal_cost:
        print(f"Optimal: {qap_instance.optimal_cost}")

    # Transfer operators
    print(f"\n--- Transfer Learning: TSP -> QAP ---")
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="qap",
        target_instance=qap_instance,
    )

    print(f"Transferred: {result.operators_transferred} operators")

    if not result.adapted_operators:
        print("ERROR: No operators were transferred!")
        return

    adapted_operators = result.adapted_operators

    # Baselines
    print(f"\n--- Baselines ---")

    # Gilmore-Lawler heuristic (1962) - domain-specific for QAP
    gl_sol = domain.gilmore_lawler_solution(qap_instance)
    gl_cost = gl_sol.cost
    print(f"Gilmore-Lawler (1962): {gl_cost}")

    greedy_sol = domain.greedy_solution(qap_instance)
    greedy_cost = greedy_sol.cost
    print(f"Greedy: {greedy_cost}")

    random_sol = domain.random_solution(qap_instance)
    random_cost = random_sol.cost
    print(f"Random: {random_cost}")

    # ILS baseline (same time budget as transfer)
    print(f"Running ILS ({args.time_limit}s)...", end=" ", flush=True)
    ils_assignment, ils_cost, ils_iters = ils_qap(
        qap_instance, domain, args.time_limit, seed=args.seed
    )
    print(f"done ({ils_iters} iterations)")
    print(f"ILS: {ils_cost}")

    # Use Gilmore-Lawler as starting point (domain-specific heuristic)
    best_baseline_sol = gl_sol
    best_baseline_cost = gl_cost
    best_baseline_name = "Gilmore-Lawler"

    print(f"\nBaseline: {best_baseline_name} ({best_baseline_cost})")

    optimal = qap_instance.optimal_cost
    if optimal:
        baseline_gap = 100 * (best_baseline_cost - optimal) / optimal
        print(f"Gap from optimal: {baseline_gap:.1f}%")

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

    # Create executor with full knowledge
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_qap_evaluate_fn(domain),
        copy_fn=qap_copy,
        operator_pheromones=operator_pheromones,
        success_frequency=success_frequency,
        global_mode=True,
        verbose=not args.quiet,
    )

    # Convert initial solution
    initial = QAPAssignment(
        assignment=list(best_baseline_sol.assignment),
        cost=best_baseline_cost,
    )

    # Execute
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=best_baseline_cost,
        time_limit=args.time_limit,
        instance=qap_instance,
    )

    best_sol = exec_result.best_solution
    final_cost = best_sol.cost

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if optimal:
        transfer_gap = 100 * (final_cost - optimal) / optimal
        baseline_gap = 100 * (best_baseline_cost - optimal) / optimal
        ils_gap = 100 * (ils_cost - optimal) / optimal
        print(f"Transfer Learning: {final_cost} (gap: {transfer_gap:.1f}%)")
        print(f"ILS:               {ils_cost} (gap: {ils_gap:.1f}%)")
        print(f"{best_baseline_name}: {best_baseline_cost} (gap: {baseline_gap:.1f}%)")
    else:
        print(f"Transfer Learning: {final_cost}")
        print(f"ILS:               {ils_cost}")
        print(f"{best_baseline_name}: {best_baseline_cost}")

    # Compare with ILS
    ils_improvement = ils_cost - final_cost
    if ils_improvement > 0:
        pct = 100 * ils_improvement / ils_cost
        print(f"\nTransfer beats ILS by: {ils_improvement} ({pct:.1f}%)")
    elif ils_improvement < 0:
        pct = 100 * (-ils_improvement) / final_cost
        print(f"\nILS beats Transfer by: {-ils_improvement} ({pct:.1f}%)")
    else:
        print(f"\nTransfer and ILS tied")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "instance": qap_instance.name,
            "n": qap_instance.n,
            "optimal": optimal,
            "transfer_cost": final_cost,
            "ils_cost": ils_cost,
            "ils_iterations": ils_iters,
            "gilmore_lawler_cost": gl_cost,
            "greedy_cost": greedy_cost,
            "random_cost": random_cost,
            "operators_transferred": result.operators_transferred,
            "time_limit": args.time_limit,
        }

        with open(output_dir / "qap_transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
