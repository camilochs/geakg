#!/usr/bin/env python3
"""Run VRP with Transfer Learning from TSP.

Uses symbolic rules extracted from a TSP AKG snapshot to guide
optimization on VRP instances.

This script contains ONLY VRP-specific logic:
- VRP baselines (nearest neighbor, Clarke-Wright)
- VRP evaluation and copy functions
- VRP instance loading

All transfer logic is in src/akg/transfer/.

Usage:
    uv run python scripts/run_vrp_transfer.py \
        --snapshot experiments/nsgge/results/.../akg_snapshot.json \
        --instance data/instances/vrp/A-n32-k5.vrp \
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
from src.geakg.transfer.adapters.vrp_adapter import VRPRoutes
from src.domains.vrp import VRPDomain, VRPSolution


# =============================================================================
# ILS BASELINE
# =============================================================================

def ils_vrp(instance, domain: VRPDomain, time_limit: float, seed: int = 42) -> tuple:
    """Iterated Local Search for VRP.

    Uses intra-route and inter-route moves with random perturbation.

    Args:
        instance: VRP instance
        domain: VRP domain
        time_limit: Time limit in seconds
        seed: Random seed

    Returns:
        (best_routes, best_cost, iterations)
    """
    random.seed(seed)

    def evaluate(routes: list) -> float:
        """Evaluate VRP solution cost."""
        sol = VRPSolution(routes=routes)
        return domain.evaluate_solution(sol, instance)

    def is_feasible(routes: list) -> bool:
        """Check capacity constraints."""
        for route in routes:
            load = sum(instance.demands[c] for c in route)
            if load > instance.capacity:
                return False
        return True

    def local_search(routes: list) -> tuple:
        """2-opt within routes + relocate between routes."""
        current = [list(r) for r in routes]
        current_cost = evaluate(current)
        improved = True

        while improved:
            improved = False

            # Intra-route 2-opt
            for r_idx, route in enumerate(current):
                if len(route) < 3:
                    continue
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        # Reverse segment
                        new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
                        new_routes = [list(r) for r in current]
                        new_routes[r_idx] = new_route
                        new_cost = evaluate(new_routes)
                        if new_cost < current_cost:
                            current = new_routes
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Inter-route relocate
            for r1_idx in range(len(current)):
                for c_idx in range(len(current[r1_idx])):
                    customer = current[r1_idx][c_idx]
                    for r2_idx in range(len(current)):
                        if r1_idx == r2_idx:
                            continue
                        # Try inserting customer into route r2
                        for pos in range(len(current[r2_idx]) + 1):
                            new_routes = [list(r) for r in current]
                            new_routes[r1_idx] = current[r1_idx][:c_idx] + current[r1_idx][c_idx+1:]
                            new_routes[r2_idx] = current[r2_idx][:pos] + [customer] + current[r2_idx][pos:]
                            # Remove empty routes
                            new_routes = [r for r in new_routes if r]
                            if is_feasible(new_routes):
                                new_cost = evaluate(new_routes)
                                if new_cost < current_cost:
                                    current = new_routes
                                    current_cost = new_cost
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return current, current_cost

    def perturb(routes: list, strength: int = 2) -> list:
        """Random relocate perturbation."""
        result = [list(r) for r in routes]
        for _ in range(strength):
            if not result:
                break
            # Pick random customer
            non_empty = [i for i, r in enumerate(result) if r]
            if not non_empty:
                break
            r1_idx = random.choice(non_empty)
            if not result[r1_idx]:
                continue
            c_idx = random.randint(0, len(result[r1_idx]) - 1)
            customer = result[r1_idx][c_idx]

            # Remove from current route
            result[r1_idx] = result[r1_idx][:c_idx] + result[r1_idx][c_idx+1:]

            # Insert into random position
            r2_idx = random.randint(0, len(result) - 1)
            pos = random.randint(0, len(result[r2_idx]))
            result[r2_idx] = result[r2_idx][:pos] + [customer] + result[r2_idx][pos:]

        # Clean empty routes
        result = [r for r in result if r]
        return result

    # Initialize with Clarke-Wright
    cw_sol = clarke_wright_savings(instance)
    current = [list(r) for r in cw_sol.routes]
    current, current_cost = local_search(current)

    best = [list(r) for r in current]
    best_cost = current_cost

    start_time = time.time()
    iterations = 0

    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(current)

        # Local search
        if is_feasible(perturbed):
            new_routes, new_cost = local_search(perturbed)

            # Accept if better (strict ILS)
            if new_cost < current_cost:
                current = new_routes
                current_cost = new_cost

                if new_cost < best_cost:
                    best = [list(r) for r in new_routes]
                    best_cost = new_cost

        iterations += 1

    return best, best_cost, iterations


# =============================================================================
# VRP-SPECIFIC BASELINES
# =============================================================================

def nearest_neighbor_vrp(instance) -> VRPSolution:
    """Simple nearest neighbor heuristic for VRP."""
    dm = instance.distance_matrix
    n = instance.n_customers
    capacity = instance.capacity
    demands = instance.demands

    routes = []
    visited = set()

    while len(visited) < n:
        route = []
        load = 0
        current = 0  # depot

        while len(visited) < n:
            best_dist = float("inf")
            best_customer = None

            for customer in range(1, n + 1):
                if customer not in visited:
                    if load + demands[customer] <= capacity:
                        dist = dm[current][customer]
                        if dist < best_dist:
                            best_dist = dist
                            best_customer = customer

            if best_customer is None:
                break

            route.append(best_customer)
            visited.add(best_customer)
            load += demands[best_customer]
            current = best_customer

        if route:
            routes.append(route)

    return VRPSolution(routes=routes)


def clarke_wright_savings(instance) -> VRPSolution:
    """Clarke-Wright savings algorithm for VRP."""
    dm = instance.distance_matrix
    n = instance.n_customers
    capacity = instance.capacity
    demands = instance.demands

    # Calculate savings
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dm[0][i] + dm[0][j] - dm[i][j]
            savings.append((s, i, j))

    savings.sort(reverse=True)

    # Initialize routes
    routes = {i: [i] for i in range(1, n + 1)}
    route_demand = {i: demands[i] for i in range(1, n + 1)}
    customer_route = {i: i for i in range(1, n + 1)}

    # Merge routes
    for s, i, j in savings:
        ri = customer_route[i]
        rj = customer_route[j]

        if ri == rj:
            continue

        if route_demand[ri] + route_demand[rj] > capacity:
            continue

        # Check if i and j are at route endpoints
        route_i = routes[ri]
        route_j = routes[rj]

        if route_i[-1] == i and route_j[0] == j:
            new_route = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            new_route = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            new_route = route_i + list(reversed(route_j))
        elif route_i[0] == i and route_j[0] == j:
            new_route = list(reversed(route_i)) + route_j
        else:
            continue

        # Merge
        routes[ri] = new_route
        route_demand[ri] = route_demand[ri] + route_demand[rj]
        del routes[rj]
        del route_demand[rj]

        for c in new_route:
            customer_route[c] = ri

    return VRPSolution(routes=list(routes.values()))


# =============================================================================
# VRP-SPECIFIC FUNCTIONS FOR SYMBOLIC EXECUTOR
# =============================================================================

def create_vrp_evaluate_fn(domain: VRPDomain):
    """Create VRP evaluation function for SymbolicExecutor."""
    def vrp_evaluate(solution, instance):
        if isinstance(solution, VRPRoutes):
            if solution.cost > 0:
                return solution.cost
            return domain.evaluate_solution(
                VRPSolution(routes=solution.routes), instance
            )
        return domain.evaluate_solution(solution, instance)
    return vrp_evaluate


def vrp_copy(solution):
    """Copy VRP solution."""
    if isinstance(solution, VRPRoutes):
        return VRPRoutes(
            routes=[list(r) for r in solution.routes],
            cost=solution.cost
        )
    return VRPSolution(routes=[list(r) for r in solution.routes])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VRP with Transfer Learning from TSP (Symbolic Rules)"
    )
    parser.add_argument("--snapshot", type=str, help="Path to TSP AKG snapshot")
    parser.add_argument(
        "--instance",
        type=str,
        default="data/instances/vrp/A-n32-k5.vrp",
        help="VRP instance file"
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
    print("VRP TRANSFER LEARNING FROM TSP (SYMBOLIC RULES)")
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

    # Load VRP instance
    print(f"\n--- Loading VRP Instance ---")
    domain = VRPDomain()
    instance_path = Path(args.instance)

    if not instance_path.exists():
        print(f"Instance not found: {instance_path}")
        return

    vrp_instance = domain.load_instance(instance_path)
    print(f"Instance: {vrp_instance.name}")
    print(f"Customers: {vrp_instance.n_customers}")
    print(f"Capacity: {vrp_instance.capacity}")
    print(f"Optimal: {vrp_instance.optimal_cost}")

    # Transfer operators
    print(f"\n--- Transfer Learning: TSP → VRP ---")
    manager = TransferManager()
    result = manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="vrp",
        target_instance=vrp_instance,
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
    nn_sol = nearest_neighbor_vrp(vrp_instance)
    nn_cost = domain.evaluate_solution(nn_sol, vrp_instance)
    nn_gap = 100 * (nn_cost - vrp_instance.optimal_cost) / vrp_instance.optimal_cost if vrp_instance.optimal_cost else 0
    print(f"Nearest Neighbor: {nn_cost:.2f} (gap: {nn_gap:.1f}%)")

    cw_sol = clarke_wright_savings(vrp_instance)
    cw_cost = domain.evaluate_solution(cw_sol, vrp_instance)
    cw_gap = 100 * (cw_cost - vrp_instance.optimal_cost) / vrp_instance.optimal_cost if vrp_instance.optimal_cost else 0
    print(f"Clarke-Wright:    {cw_cost:.2f} (gap: {cw_gap:.1f}%)")

    # ILS baseline (same time budget as transfer)
    print(f"\nRunning ILS ({args.time_limit}s)...", end=" ", flush=True)
    ils_routes, ils_cost, ils_iters = ils_vrp(
        vrp_instance, domain, args.time_limit, seed=42
    )
    print(f"done ({ils_iters} iterations)")
    ils_gap = 100 * (ils_cost - vrp_instance.optimal_cost) / vrp_instance.optimal_cost if vrp_instance.optimal_cost else 0
    print(f"ILS: {ils_cost:.2f} (gap: {ils_gap:.1f}%)")

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

    # Create executor with VRP-specific functions
    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=create_vrp_evaluate_fn(domain),
        copy_fn=vrp_copy,
        operator_pheromones=operator_pheromones,
        success_frequency=success_frequency,
        global_mode=True,
        verbose=not args.quiet,
    )

    # Convert initial solution
    initial = VRPRoutes(
        routes=[list(r) for r in cw_sol.routes],
        cost=cw_cost,
    )

    # Execute
    exec_result = executor.execute(
        operators=adapted_operators,
        initial_solution=initial,
        initial_cost=cw_cost,
        time_limit=args.time_limit,
        instance=vrp_instance,
    )

    best_sol = exec_result.best_solution
    history = exec_result.history

    final_cost = best_sol.cost

    # Final results
    optimal = vrp_instance.optimal_cost or cw_cost
    transfer_gap = 100 * (final_cost - optimal) / optimal

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Transfer Learning: {final_cost:.2f} (gap: {transfer_gap:.1f}%)")
    print(f"ILS:               {ils_cost:.2f} (gap: {ils_gap:.1f}%)")
    print(f"Clarke-Wright:     {cw_cost:.2f} (gap: {cw_gap:.1f}%)")
    print(f"Nearest Neighbor:  {nn_cost:.2f} (gap: {nn_gap:.1f}%)")

    # Compare with ILS
    ils_improvement = ils_cost - final_cost
    if ils_improvement > 0:
        pct = 100 * ils_improvement / ils_cost
        print(f"\nTransfer beats ILS by: {ils_improvement:.2f} ({pct:.1f}%)")
    elif ils_improvement < 0:
        pct = 100 * (-ils_improvement) / final_cost
        print(f"\nILS beats Transfer by: {-ils_improvement:.2f} ({pct:.1f}%)")
    else:
        print(f"\nTransfer and ILS tied")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "instance": vrp_instance.name,
            "optimal": vrp_instance.optimal_cost,
            "transfer_cost": final_cost,
            "transfer_gap": transfer_gap,
            "ils_cost": ils_cost,
            "ils_gap": ils_gap,
            "ils_iterations": ils_iters,
            "cw_cost": cw_cost,
            "cw_gap": cw_gap,
            "nn_cost": nn_cost,
            "nn_gap": nn_gap,
            "operators_transferred": result.operators_transferred,
            "time_limit": args.time_limit,
            "history": history,
            "best_routes": best_sol.routes,
        }

        with open(output_dir / "vrp_transfer_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
