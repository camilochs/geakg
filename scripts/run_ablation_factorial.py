#!/usr/bin/env python3
"""Factorial ablation on a FROZEN L1 pool (offline, no LLM).

Headline axis (KBS R1.1, the demanded & missing ablation):
  ontology-constrained topology  vs  unrestricted (fully-connected) traversal.
Second axis (KBS R4.3 component isolation):
  L2 learned role pheromones  ON (alpha=2)  vs  OFF (alpha=0).

To isolate the STRUCTURAL constraint cleanly, the runtime ontology mechanisms
(conditional-edge boosts and incompatibility tracking) are held OFF in all cells,
so the only thing distinguishing constrained from unrestricted is which role
transitions are permitted by the MetaGraph.

A one-time warmup validates the operator set (drops operators that timeout/crash)
so every cell runs on the IDENTICAL set -- no global-disabling confound, no repeated
timeouts. Pruning is off in all cells. Optimization heuristics are stochastic, so we
report mean +/- std over seeds; the qualitative ordering (who wins) is the signal.

Usage:
  uv run python scripts/run_ablation_factorial.py \
    --instances data/instances/tsp/berlin52.tsp data/instances/tsp/eil51.tsp \
    --seeds 42 43 44 45 46 --iterations 25 --n-ants 12 \
    --output results/ablation/factorial_tsp.json
  uv run python scripts/run_ablation_factorial.py --quick     # tiny smoke config
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path

from loguru import logger


def _make_unrestricted(meta_graph, weight: float = 0.5):
    """Add edges for every ordered role pair so traversal is unconstrained."""
    from src.geakg.layers.l0.metagraph import MetaEdge

    roles = list(meta_graph.nodes.keys())
    for s in roles:
        for t in roles:
            if s != t and not meta_graph.has_edge(s, t):
                meta_graph.add_edge(MetaEdge(source=s, target=t, weight=weight))
    return meta_graph


def build_selector(pool_path: str, l2_on: bool, topology: str, n_ants: int,
                   exclude_ops: set | None = None):
    """Build a MetaACOSelector over the frozen pool with the ablation toggles.

    topology: "constrained" (RoleSchema-valid transitions only) or
              "unrestricted" (fully-connected role graph).
    exclude_ops: operator names removed before building, so every cell runs on the
                 SAME validated operator set.
    """
    from src.geakg.aco import MetaACOConfig, MetaACOSelector, OperatorMode
    from src.geakg.bindings import BindingRegistry
    from src.geakg.layers.l1 import OperatorPool
    from src.geakg.layers.l1.hook import L0SynthesisHook
    from src.geakg.layers.l0.patterns import create_hybrid_meta_graph
    from src.geakg.layers.l0.metagraph import InstantiatedGraph

    pool = OperatorPool.load(pool_path)
    for op in (exclude_ops or set()):
        pool.remove_operator(op)
    BindingRegistry.reset()
    bindings = BindingRegistry().get_domain("tsp")
    hook = L0SynthesisHook(pool)
    hook.register_operators_to_bindings(bindings)

    meta_graph = create_hybrid_meta_graph()
    if topology == "unrestricted":
        _make_unrestricted(meta_graph)
    instantiated = InstantiatedGraph(meta_graph, bindings)

    cfg = MetaACOConfig(
        n_ants=n_ants,
        operator_mode=OperatorMode.STATIC,
        alpha=2.0 if l2_on else 0.0,
        enable_conditions=False,            # held off: isolate the structural constraint
        enable_incompatibility_tracking=False,
        enable_synthesis=False,
        enable_pruning=False,
    )
    selector = MetaACOSelector(instantiated, cfg, synthesis_hook=hook)
    return selector, hook, pool, meta_graph


def warmup_validate(pool_path, instances, domain_config, n_ants, iterations=10):
    """One-time pass to find operators that fail (timeout/crash); returns their names."""
    from src.geakg.execution import (
        evaluate_operator_path_with_stats,
        reset_disabled_operators,
        get_disabled_operators,
    )
    random.seed(0)
    reset_disabled_operators()
    selector, hook, _pool, _mg = build_selector(pool_path, True, "constrained", n_ants)
    for _ in range(iterations):
        for _ in range(n_ants):
            ant = selector.construct_solution()
            if not getattr(ant, "operator_path", None):
                continue
            for inst in instances:
                try:
                    evaluate_operator_path_with_stats(
                        ant.operator_path, inst.instance_data, domain_config, hook)
                except Exception:
                    pass
    return set(get_disabled_operators())


def run_cell(pool_path, instances, domain_config, l2_on, topology,
             n_ants, iterations, seed, exclude_ops=None):
    """Run one ablation cell; return best mean-gap (%) across instances."""
    from src.geakg.execution import (
        evaluate_operator_path_with_stats,
        reset_disabled_operators,
    )

    random.seed(seed)
    reset_disabled_operators()
    selector, hook, _pool, _mg = build_selector(
        pool_path, l2_on, topology, n_ants, exclude_ops=exclude_ops)

    best = float("inf")
    all_gaps = []  # quality of every ant (mean controls for exploration-breadth confound)
    for _ in range(iterations):
        for _ in range(n_ants):
            ant = selector.construct_solution()
            if not getattr(ant, "operator_path", None):
                continue
            gaps = []
            for inst in instances:
                fitness, _deltas = evaluate_operator_path_with_stats(
                    ant.operator_path, inst.instance_data, domain_config, hook
                )
                if inst.optimal:
                    gaps.append(100.0 * (fitness - inst.optimal) / inst.optimal)
                else:
                    gaps.append(fitness / inst.dimension)
            avg_gap = sum(gaps) / len(gaps)
            selector.update_pheromones_for_path(ant.role_path, avg_gap, ant.operator_path)
            best = min(best, avg_gap)
            all_gaps.append(avg_gap)
    return {"best": best, "mean": statistics.mean(all_gaps) if all_gaps else float("inf"),
            "n_ants_eval": len(all_gaps)}


def main():
    ap = argparse.ArgumentParser(description="Factorial ablation (L2 x topology) on frozen pool")
    ap.add_argument("--pool", default="pools/tsp_pool.json")
    ap.add_argument("--instances", nargs="+",
                    default=["data/instances/tsp/berlin52.tsp", "data/instances/tsp/eil51.tsp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--n-ants", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="tiny smoke config")
    ap.add_argument("--output", default="results/ablation/factorial_tsp.json")
    args = ap.parse_args()

    if args.quick:
        args.instances = args.instances[:1]
        args.seeds = [42, 43]
        args.iterations = 8
        args.n_ants = 8

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from src.geakg.instance_pool import InstancePool
    from src.domains import get_domain_config

    ip = InstancePool(domain="tsp")
    ip.load_instances_from_files(args.instances)
    instances = ip.instances
    if not instances:
        print("ERROR: no instances loaded")
        sys.exit(1)
    domain_config = get_domain_config("tsp")

    print("=" * 64)
    print("FACTORIAL ABLATION  L2 x topology  (frozen pool, offline, no LLM)")
    print("=" * 64)
    print(f"pool={args.pool}")
    print(f"instances={[i.instance_id for i in instances]}")
    print(f"seeds={args.seeds} iterations={args.iterations} n_ants={args.n_ants}")
    print()

    print("--- warmup: validating operator set (one-time) ---")
    exclude_ops = warmup_validate(args.pool, instances, domain_config, args.n_ants)
    print(f"excluded {len(exclude_ops)} failing operators: {sorted(exclude_ops)}\n")

    # Report topology sizes for the paper (constrained vs unrestricted edge counts).
    _, _, _, mg_c = build_selector(args.pool, True, "constrained", args.n_ants, exclude_ops)
    _, _, _, mg_u = build_selector(args.pool, True, "unrestricted", args.n_ants, exclude_ops)
    print(f"topology edges: constrained={len(mg_c.edges)}  unrestricted={len(mg_u.edges)}  "
          f"(roles={len(mg_c.nodes)})\n")

    cells = list(product([True, False], ["constrained", "unrestricted"]))
    results = []
    for l2_on, topo in cells:
        for seed in args.seeds:
            t0 = time.time()
            cell = run_cell(args.pool, instances, domain_config, l2_on, topo,
                            args.n_ants, args.iterations, seed, exclude_ops=exclude_ops)
            dt = time.time() - t0
            results.append({"L2": l2_on, "topology": topo, "seed": seed,
                            "best_gap": cell["best"], "mean_gap": cell["mean"],
                            "secs": round(dt, 1)})
            print(f"  L2={int(l2_on)} topo={topo:12} seed={seed}: "
                  f"best={cell['best']:.3f}%  mean={cell['mean']:.3f}%  ({dt:.1f}s)")

    def agg(metric, l2, topo):
        v = [r[metric] for r in results if r["L2"] == l2 and r["topology"] == topo]
        if not v:
            return (None, None, 0)
        return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0, len(v))

    print("\n" + "=" * 64)
    print("SUMMARY (mean over seeds; lower=better). best=luckiest sample (rewards")
    print("exploration breadth); mean=avg solution quality (fair across topologies)")
    print("=" * 64)
    summary = {}
    for l2_on, topo in cells:
        bm, bs, n = agg("best_gap", l2_on, topo)
        mm, ms, _ = agg("mean_gap", l2_on, topo)
        key = f"L2={int(l2_on)},{topo}"
        summary[key] = {"best_mean": bm, "best_std": bs, "mean_mean": mm, "mean_std": ms, "n": n}
        print(f"  {key:24}  best {bm:6.3f}±{bs:.3f}   mean {mm:6.3f}±{ms:.3f}  (n={n})")

    print("\nisolation on MEAN gap (diversity-controlled; + => mechanism helps):")
    for l2 in (True, False):
        c = agg("mean_gap", l2, "constrained")[0]
        u = agg("mean_gap", l2, "unrestricted")[0]
        if c is not None and u is not None:
            verdict = "constrained better" if u > c else "unrestricted better"
            print(f"  unrestricted - constrained (L2={int(l2)}): {u - c:+.3f} pp  ({verdict})")
    for topo in ("constrained", "unrestricted"):
        on = agg("mean_gap", True, topo)[0]
        off = agg("mean_gap", False, topo)[0]
        if on is not None and off is not None:
            verdict = "L2 helps" if off > on else "L2 hurts"
            print(f"  remove L2 ({topo}): {off - on:+.3f} pp  ({verdict})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "topology_edges": {"constrained": len(mg_c.edges),
                                      "unrestricted": len(mg_u.edges)},
                   "results": results, "summary": summary}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
