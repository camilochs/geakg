#!/usr/bin/env python3
"""L0 ontology ablation (faithful, token-free): does restricting role transitions
to RoleSchema-valid category transitions help, or is the ontology decorative?

Runs the REAL offline ACO (MetaACOSelector) over the REAL operator pool, on the
same TSP instances, comparing two L0 topologies:
  CONSTRAINED  : create_default_metagraph_for_pool  (only valid category transitions)
  UNRESTRICTED : same graph + every role->role edge  (ontology constraint removed)
Both learn L2 from scratch; lower average gap is better. CONSTRAINED <= UNRESTRICTED
means the ontology constraint carries design value (R1.1). No LLM calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics as st
import time
from pathlib import Path

from src.geakg.layers.l1.pool import OperatorPool
from src.geakg.instance_pool import InstancePool
from src.domains import get_domain_config
from src.geakg.bindings import BindingRegistry
from src.geakg.conditions import ExecutionContext
from src.geakg.execution import evaluate_operator_path_with_stats
from src.geakg.layers.l1.hook import L0SynthesisHook
from src.geakg.layers.l0.topology_generator import create_default_metagraph_for_pool
from src.geakg.layers.l0.metagraph import InstantiatedGraph, MetaEdge
from src.geakg.aco import MetaACOConfig, MetaACOSelector, OperatorMode


def make_unrestricted(constrained, roles):
    mg = copy.deepcopy(constrained)
    added = 0
    for a in roles:
        for b in roles:
            if a == b:
                continue
            if mg.get_edge(a, b) is None:
                mg.add_edge(MetaEdge(source=a, target=b, weight=0.5))
                added += 1
    return mg, added


def run_aco(meta_graph, bindings, hook, instance_pool, domain_config, aco_cfg, timeout, seed):
    random.seed(seed)
    instantiated = InstantiatedGraph(meta_graph, bindings)
    selector = MetaACOSelector(instantiated, aco_cfg, synthesis_hook=hook)
    start = time.time()
    best = float("inf")
    stagn = 0
    dim = instance_pool.instances[0].dimension
    while time.time() - start < timeout:
        ibest = float("inf")
        iant = None
        for _ in range(aco_cfg.n_ants):
            ant = selector.construct_solution(problem_size=dim)
            if not ant.operator_path:
                continue
            gaps = []
            for inst in instance_pool.instances:
                fit, _ = evaluate_operator_path_with_stats(
                    ant.operator_path, inst.instance_data, domain_config, hook)
                gaps.append(100 * (fit - inst.optimal) / inst.optimal if inst.optimal else fit / inst.dimension)
            ag = sum(gaps) / len(gaps)
            ant.gap = ag
            if ag < ibest:
                ibest = ag
                iant = ant
        selector.set_execution_context(ExecutionContext(
            generations_without_improvement=stagn, population_diversity=0.5,
            current_fitness=ibest, best_fitness=best))
        if iant and ibest < best:
            best = ibest
            stagn = 0
            selector.record_successful_path(iant.role_path, iant.operator_path)
        else:
            stagn += 1
        if iant:
            selector.update_pheromones_for_path(iant.role_path, ibest, operator_path=iant.operator_path)
            selector.update_operator_pheromones(iant, ibest)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experiments/iterative/20260125_121313_iterative/refined_pool.json")
    ap.add_argument("--instances", nargs="+",
                    default=["data/instances/tsp/berlin52.tsp", "data/instances/tsp/kroA100.tsp",
                             "data/instances/tsp/ch150.tsp", "data/instances/tsp/pr226.tsp"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--out", default="results/ablation/L0_ontology_ablation.json")
    args = ap.parse_args()

    pool = OperatorPool.load(args.pool)
    instance_pool = InstancePool(domain="tsp")
    instance_pool.load_instances_from_files(args.instances)
    domain_config = get_domain_config("tsp")
    BindingRegistry.reset()
    bindings = BindingRegistry().get_domain("tsp")
    hook = L0SynthesisHook(pool)
    hook.register_operators_to_bindings(bindings)

    constrained = create_default_metagraph_for_pool(pool)
    roles = sorted({r for e in constrained.edges for r in e})
    unrestricted, added = make_unrestricted(constrained, roles)
    aco_cfg = MetaACOConfig(operator_mode=OperatorMode.DYNAMIC, enable_synthesis=False,
                            enable_conditions=True, enable_incompatibility_tracking=True)
    print(f"roles={len(roles)}  constrained edges={len(constrained.edges)}  "
          f"unrestricted edges={len(unrestricted.edges)} (+{added})  n_ants={aco_cfg.n_ants}")

    rows = []
    for seed in range(args.seeds):
        gc = run_aco(constrained, bindings, hook, instance_pool, domain_config, aco_cfg, args.time, seed)
        gu = run_aco(unrestricted, bindings, hook, instance_pool, domain_config, aco_cfg, args.time, seed)
        rows.append({"seed": seed, "constrained": gc, "unrestricted": gu})
        print(f"seed {seed}: constrained {gc:.2f}%  unrestricted {gu:.2f}%  "
              f"{'C<=U' if gc <= gu else 'U<C'}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    C = [r["constrained"] for r in rows]
    U = [r["unrestricted"] for r in rows]
    print(f"\nCONSTRAINED {st.mean(C):.2f}±{st.pstdev(C):.2f}  vs  "
          f"UNRESTRICTED {st.mean(U):.2f}±{st.pstdev(U):.2f}   "
          f"(constrained better on {sum(c<=u for c,u in zip(C,U))}/{len(C)} seeds)")


if __name__ == "__main__":
    main()
