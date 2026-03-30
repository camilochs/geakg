#!/usr/bin/env python3
"""Run symbolic executor on TSP using snapshot knowledge.

Uses the same SymbolicRuleEngine and SymbolicExecutor from the transfer module,
but without domain adaptation (TSP is the source domain).

Usage:
    python scripts/run_symbolic_tsp.py data/instances/tsp/berlin52.tsp -t 30
    python scripts/run_symbolic_tsp.py data/instances/tsp/berlin52.tsp -t 60 --snapshot path/to/snapshot.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

# Add src to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from akg.transfer.symbolic_rules import extract_symbolic_rules, print_symbolic_rules
from akg.transfer.symbolic_executor import SymbolicExecutor, ExecutionResult


class TSPOperator:
    """Wrapper to make pool operators compatible with SymbolicExecutor."""
    def __init__(self, name: str, role: str, fn: callable):
        self._name = name
        self._role = role
        self._fn = fn

    @property
    def name(self):
        return self._name

    @property
    def operator_id(self):
        return self._name

    @property
    def role(self):
        return self._role

    @property
    def adapted_fn(self) -> callable:
        return self._fn


# Use the full TSP context with delta(), neighbors(), etc.
from akg.contexts.tsp import TSPContext


def load_tsplib(filepath: str) -> tuple[list[list[float]], float | None]:
    """Load TSP instance from TSPLIB format."""
    with open(filepath) as f:
        lines = f.read().strip().split("\n")

    edge_type, optimal, i = "EUC_2D", None, 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("EDGE_WEIGHT_TYPE"):
            edge_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            i += 1
            break
        elif "optimal" in line.lower():
            import re
            m = re.search(r"optimal[=:\s]+?(\d+(?:\.\d+)?)", line.lower())
            if m:
                optimal = float(m.group(1))
        i += 1

    coords = []
    while i < len(lines) and not lines[i].strip().startswith(("EOF", "DISPLAY")):
        parts = lines[i].split()
        if len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))
        i += 1

    n = len(coords)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]
                d = math.sqrt(dx * dx + dy * dy)
                dm[i][j] = math.ceil(d) if edge_type == "CEIL_2D" else d
    return dm, optimal


def load_snapshot(snapshot_path: Path) -> dict:
    """Load AKG snapshot."""
    with open(snapshot_path) as f:
        return json.load(f)


def load_pool(pool_path: Path) -> dict:
    """Load operator pool."""
    with open(pool_path) as f:
        return json.load(f)


def compile_operators(pool: dict, ctx: TSPContext) -> list[TSPOperator]:
    """Compile operators from pool into callable functions."""
    operators = []

    for role, ops_list in pool.get("operators_by_role", {}).items():
        for op_data in ops_list:
            code = op_data["code"]
            local_ns = {}
            exec(code, local_ns)

            # Find the function
            fn = None
            for name, obj in local_ns.items():
                if callable(obj) and not name.startswith("_"):
                    fn = obj
                    break

            if fn:
                # Wrap to inject ctx
                def make_wrapper(f, c):
                    def wrapper(solution, instance=None):
                        return f(solution, c)
                    return wrapper

                operators.append(TSPOperator(
                    name=op_data["name"],
                    role=role,
                    fn=make_wrapper(fn, ctx),
                ))

    return operators


def run_symbolic_with_restarts(
    executor: SymbolicExecutor,
    operators: list[TSPOperator],
    n: int,
    ctx: TSPContext,
    time_limit: float,
    verbose: bool = True,
) -> tuple[list[int], float, int]:
    """Run symbolic executor with multistart."""
    import time

    best_sol, best_cost = None, float("inf")
    start = time.time()
    restarts = 0
    total_improvements = 0

    while time.time() - start < time_limit:
        restarts += 1
        remaining_time = time_limit - (time.time() - start)
        if remaining_time <= 0:
            break

        # Generate random initial solution
        initial = list(range(n))
        random.shuffle(initial)
        initial_cost = ctx.evaluate(initial)

        # Run symbolic executor for a portion of remaining time
        run_time = min(remaining_time, max(1.0, remaining_time / 10))

        # Reset rule engine state
        executor.rule_engine.state.phase = executor.rule_engine.state.phase.__class__.CONSTRUCTION
        executor.rule_engine.state.iterations_since_improvement = 0
        executor.rule_engine.state.escalations_at_current_level = 0

        result = executor.execute(
            operators=operators,
            initial_solution=initial,
            initial_cost=initial_cost,
            time_limit=run_time,
            instance=None,
        )

        if result.best_cost < best_cost:
            best_sol = result.best_solution
            best_cost = result.best_cost
            total_improvements += result.improvements

    return best_sol, best_cost, restarts


def main():
    parser = argparse.ArgumentParser(description="Run symbolic executor on TSP")
    parser.add_argument("instance", help="Path to TSP instance")
    parser.add_argument("-t", "--timeout", type=float, default=30.0, help="Time limit in seconds")
    parser.add_argument("--snapshot", type=str, help="Path to AKG snapshot (auto-detect if not specified)")
    parser.add_argument("--pool", type=str, help="Path to operator pool (auto-detect if not specified)")
    parser.add_argument("--optimal", type=float, help="Known optimal value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--show-rules", action="store_true", help="Print extracted rules")
    args = parser.parse_args()

    # Find snapshot and pool
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        pool_path = Path(args.pool) if args.pool else snapshot_path.parent / "refined_pool.json"
    else:
        # Auto-detect: find most recent experiment
        exp_dir = Path("experiments/iterative")
        if exp_dir.exists():
            subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
            if subdirs:
                snapshot_path = subdirs[0] / "akg_snapshot.json"
                pool_path = subdirs[0] / "refined_pool.json"
            else:
                print("No experiments found. Run training first.")
                return
        else:
            print("No experiments directory. Run training first.")
            return

    if not snapshot_path.exists():
        print(f"Snapshot not found: {snapshot_path}")
        return
    if not pool_path.exists():
        print(f"Pool not found: {pool_path}")
        return

    # Load everything
    print(f"Loading snapshot: {snapshot_path}")
    print(f"Loading pool: {pool_path}")
    snapshot = load_snapshot(snapshot_path)
    pool = load_pool(pool_path)

    print(f"Loading instance: {args.instance}")
    dm, opt = load_tsplib(args.instance)
    opt = args.optimal or opt
    n = len(dm)

    print()
    print(f"Instance: {Path(args.instance).name}, n={n}, optimal={opt}")
    print(f"Snapshot gap: {snapshot.get('best_gap', 'N/A'):.2f}%")
    print(f"Operators: {sum(len(ops) for ops in pool.get('operators_by_role', {}).values())}")
    print()

    # Extract symbolic rules from snapshot
    rule_engine = extract_symbolic_rules(snapshot)

    if args.show_rules:
        print_symbolic_rules(rule_engine)
        print()

    # Compile operators
    ctx = TSPContext(dm)
    operators = compile_operators(pool, ctx)

    # Get operator pheromones
    pheromones = snapshot.get("pheromones", {})
    operator_pheromones = pheromones.get("operator_level", {})

    # Create executor
    def evaluate_fn(solution, instance):
        return ctx.evaluate(solution)

    def copy_fn(solution):
        return solution[:]

    executor = SymbolicExecutor(
        rule_engine=rule_engine,
        evaluate_fn=evaluate_fn,
        copy_fn=copy_fn,
        operator_pheromones=operator_pheromones,
        global_mode=False,
        verbose=not args.quiet,
    )

    # Run
    random.seed(args.seed)
    print(f"Running symbolic search for {args.timeout}s...")
    print()

    best_sol, best_cost, restarts = run_symbolic_with_restarts(
        executor=executor,
        operators=operators,
        n=n,
        ctx=ctx,
        time_limit=args.timeout,
        verbose=not args.quiet,
    )

    print()
    print(f"Restarts: {restarts}")
    print(f"Best cost: {best_cost:.2f}")
    if opt:
        gap = 100 * (best_cost - opt) / opt
        print(f"Gap: {gap:.2f}%")


if __name__ == "__main__":
    main()
