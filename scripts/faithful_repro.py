#!/usr/bin/env python3
"""Faithfulness check: reproduce the trained GEAKG best path OFFLINE.

Loads the FROZEN operators from a session's best_program.py and runs its
BEST_OPERATOR_PATH using the REAL GEAKG context (domain_config.create_context,
which provides neighbors()/delta() that best_program's own minimal TSPContext
lacks). The live wrapper `llamea_live_ls` (no frozen code) is aliased to the
evolved LS-large operator it produced (`ls_intensify_large_adaptive_lkh_d4b6d5`).

If the gap reproduces the paper's reported range, the frozen artifact + real
runtime are a faithful, API-free basis for the ablation.

  uv run python scripts/faithful_repro.py \
    --session experiments/iterative/20260125_121313_iterative \
    --instances data/instances/tsp/berlin52.tsp data/instances/tsp/eil51.tsp \
    --runs 15
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

from loguru import logger


def load_frozen_operators(session: Path):
    """Exec best_program.py with the live-wrapper aliased to its evolved op."""
    src = (session / "best_program.py").read_text()
    # best_program references llamea_live_ls without defining it (it was a live
    # wrapper). Freeze it to the evolved LS-large operator present in the file.
    inject = ("llamea_live_ls = ls_intensify_large_adaptive_lkh_d4b6d5  "
              "# offline freeze of the live LLaMEA wrapper\n")
    src = src.replace("OPERATORS = {", inject + "OPERATORS = {", 1)
    ns: dict = {"__name__": "_frozen"}
    exec(compile(src, str(session / "best_program.py"), "exec"), ns)
    return ns["OPERATORS"], ns["BEST_OPERATOR_PATH"], ns["BEST_ROLE_PATH"]


def run_path(ctx, n, operators, path, n_runs, seed0=42):
    best = float("inf")
    skipped = {}
    for r in range(n_runs):
        random.seed(seed0 + r)
        sol = list(range(n))
        random.shuffle(sol)
        for op_name in path:
            fn = operators.get(op_name)
            if fn is None:
                skipped[op_name] = skipped.get(op_name, 0) + 1
                continue
            try:
                out = fn(sol, ctx)
                if isinstance(out, list) and ctx.valid(out):
                    sol = out
            except Exception as e:
                skipped[op_name] = skipped.get(op_name, 0) + 1
        best = min(best, ctx.evaluate(sol))
    return best, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="experiments/iterative/20260125_121313_iterative")
    ap.add_argument("--instances", nargs="+",
                    default=["data/instances/tsp/berlin52.tsp", "data/instances/tsp/eil51.tsp"])
    ap.add_argument("--runs", type=int, default=15)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    session = Path(args.session)
    operators, best_path, best_role_path = load_frozen_operators(session)
    print(f"session: {session.name}")
    print(f"frozen operators: {sorted(operators.keys())}")
    print(f"best path ({len(best_path)} ops): {best_path}\n")

    from src.domains import get_domain_config
    from src.geakg.instance_pool import InstancePool

    dc = get_domain_config("tsp")
    ip = InstancePool(domain="tsp")
    ip.load_instances_from_files(args.instances)

    # verify the real ctx exposes what the evolved operators need
    sample_ctx = dc.create_context(ip.instances[0].instance_data)
    have = [m for m in ("evaluate", "valid", "neighbors", "delta", "cost")
            if hasattr(sample_ctx, m)]
    print(f"real ctx methods present: {have}\n")

    print(f"{'instance':12} {'n':>5} {'opt':>10} {'best':>12} {'gap%':>8}")
    for inst in ip.instances:
        ctx = dc.create_context(inst.instance_data)
        n = inst.dimension
        best, skipped = run_path(ctx, n, operators, best_path, args.runs)
        gap = 100.0 * (best - inst.optimal) / inst.optimal if inst.optimal else float("nan")
        print(f"{inst.instance_id:12} {n:>5} {str(inst.optimal):>10} {best:>12.2f} {gap:>8.3f}")
        if skipped:
            print(f"   (skipped: {skipped})")


if __name__ == "__main__":
    main()
