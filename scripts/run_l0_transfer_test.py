#!/usr/bin/env python3
"""L0 ontology TRANSFER test (the definitive ablation): does the ontology-respecting
topology produce a snapshot that TRANSFERS better than an unrestricted one?

Trains two snapshots on TSP with the REAL DYNAMIC ACO (token-free):
  CONSTRAINED  : only valid category transitions
  UNRESTRICTED : every role->role edge (ontology removed)
then transfers each zero-shot to JSSP and compares makespan gaps. The ontology's
claimed value is transfer (valid transitions are semantically meaningful and
generalize; unrestricted shortcuts overfit to TSP), so this is the fair test.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
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

from run_l0_ontology_ablation import make_unrestricted
from run_jssp_ablation_multiseed import symbolic_run
from src.geakg.transfer.transfer_manager import TransferManager
from src.domains.jssp import JSSPDomain
from src.geakg.transfer import extract_symbolic_rules
from run_jssp_transfer import spt_dispatch, lpt_dispatch


def train(meta_graph, bindings, hook, instance_pool, domain_config, aco_cfg, timeout, seed):
    random.seed(seed)
    selector = MetaACOSelector(InstantiatedGraph(meta_graph, bindings), aco_cfg, synthesis_hook=hook)
    start = time.time(); best = float("inf"); stagn = 0
    dim = instance_pool.instances[0].dimension
    while time.time() - start < timeout:
        ibest = float("inf"); iant = None
        for _ in range(aco_cfg.n_ants):
            ant = selector.construct_solution(problem_size=dim)
            if not ant.operator_path:
                continue
            gaps = []
            for inst in instance_pool.instances:
                fit, _ = evaluate_operator_path_with_stats(ant.operator_path, inst.instance_data, domain_config, hook)
                gaps.append(100 * (fit - inst.optimal) / inst.optimal if inst.optimal else fit / inst.dimension)
            ag = sum(gaps) / len(gaps); ant.gap = ag
            if ag < ibest:
                ibest = ag; iant = ant
        selector.set_execution_context(ExecutionContext(generations_without_improvement=stagn,
            population_diversity=0.5, current_fitness=ibest, best_fitness=best))
        if iant and ibest < best:
            best = ibest; stagn = 0; selector.record_successful_path(iant.role_path, iant.operator_path)
        else:
            stagn += 1
        if iant:
            selector.update_pheromones_for_path(iant.role_path, ibest, operator_path=iant.operator_path)
            selector.update_operator_pheromones(iant, ibest)
    return best, selector


def save_snapshot(selector, meta_graph, out_dir, pool_path, best_gap):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    snap = {
        "domain": "tsp", "best_gap": best_gap,
        "pheromones": {
            "role_level": {f"{s}->{t}": tau for (s, t), tau in selector.pheromones.items()},
            "operator_level": {f"{r}:{op}": tau for (r, op), tau in selector.get_operator_pheromones().items()},
        },
        "metagraph": {"name": meta_graph.name,
                      "edges": [{"source": s, "target": t, "weight": 0.5} for (s, t) in meta_graph.edges]},
        "successful_paths": [{"operators": p} for p in (selector.get_successful_paths() or [])][:20],
    }
    json.dump(snap, open(out / "akg_snapshot.json", "w"), indent=1)
    shutil.copy(pool_path, out / "refined_pool.json")
    return out / "akg_snapshot.json"


def transfer_to_jssp(snap_path, dom, inst, seed, time_limit):
    snapshot = json.load(open(snap_path))
    res = TransferManager().transfer_from_akg(source_snapshot=str(snap_path), target_domain="jssp", target_instance=inst)
    ops = res.adapted_operators
    rule_engine = extract_symbolic_rules(snapshot)
    pher = snapshot["pheromones"]["operator_level"]
    sf = {}
    for p in snapshot.get("successful_paths", []):
        for op in p.get("operators", []):
            sf[op] = sf.get(op, 0) + 1
    spt = spt_dispatch(inst); spt_mk = dom.evaluate_solution(spt, inst)
    lpt = lpt_dispatch(inst); lpt_mk = dom.evaluate_solution(lpt, inst)
    base_sol, base_mk = (spt, spt_mk) if spt_mk <= lpt_mk else (lpt, lpt_mk)
    mk = symbolic_run(dom, inst, snapshot, ops, rule_engine, pher, sf, base_sol, base_mk, time_limit, seed)
    opt = inst.optimal_makespan
    return 100 * (mk - opt) / opt if opt else mk / inst.dimension, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experiments/iterative/20260125_121313_iterative/refined_pool.json")
    ap.add_argument("--train-instances", nargs="+",
                    default=["data/instances/tsp/berlin52.tsp", "data/instances/tsp/kroA100.tsp"])
    ap.add_argument("--train-time", type=float, default=30.0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--jssp", nargs="+", default=["ft06", "ft10", "la21", "la31", "abz7"])
    ap.add_argument("--jssp-seeds", type=int, default=5)
    ap.add_argument("--jssp-time", type=float, default=30.0)
    ap.add_argument("--out", default="results/ablation/L0_transfer_test.json")
    args = ap.parse_args()

    pool = OperatorPool.load(args.pool)
    instance_pool = InstancePool(domain="tsp")
    instance_pool.load_instances_from_files(args.train_instances)
    domain_config = get_domain_config("tsp")
    BindingRegistry.reset()
    bindings = BindingRegistry().get_domain("tsp")
    hook = L0SynthesisHook(pool); hook.register_operators_to_bindings(bindings)
    constrained = create_default_metagraph_for_pool(pool)
    roles = sorted({r for e in constrained.edges for r in e})
    unrestricted, _ = make_unrestricted(constrained, roles)
    aco_cfg = MetaACOConfig(operator_mode=OperatorMode.DYNAMIC, enable_synthesis=False,
                            enable_conditions=True, enable_incompatibility_tracking=True)

    print(f"Training CONSTRAINED ({len(constrained.edges)} edges) and UNRESTRICTED "
          f"({len(unrestricted.edges)} edges) snapshots on TSP ({args.train_time}s each)...")
    gc, sel_c = train(constrained, bindings, hook, instance_pool, domain_config, aco_cfg, args.train_time, args.train_seed)
    snap_c = save_snapshot(sel_c, constrained, "results/ablation/snap_constrained", args.pool, gc)
    hook2 = L0SynthesisHook(pool); BindingRegistry.reset(); bindings2 = BindingRegistry().get_domain("tsp")
    hook2.register_operators_to_bindings(bindings2)
    gu, sel_u = train(unrestricted, bindings2, hook2, instance_pool, domain_config, aco_cfg, args.train_time, args.train_seed)
    snap_u = save_snapshot(sel_u, unrestricted, "results/ablation/snap_unrestricted", args.pool, gu)
    print(f"  trained TSP gaps: constrained {gc:.2f}%  unrestricted {gu:.2f}%")

    dom = JSSPDomain()
    rows = []
    print(f"\nTransfer to JSSP ({args.jssp_seeds} seeds, {args.jssp_time}s):")
    print(f"{'inst':6}{'constrained':>16}{'unrestricted':>16}  winner")
    for name in args.jssp:
        inst = dom.load_instance(Path(f"data/instances/jssp/parsed/{name}.txt"))
        gc_list, gu_list = [], []
        for s in range(args.jssp_seeds):
            gc_j, _ = transfer_to_jssp(snap_c, dom, inst, s, args.jssp_time)
            gu_j, _ = transfer_to_jssp(snap_u, dom, inst, s, args.jssp_time)
            gc_list.append(gc_j); gu_list.append(gu_j)
        cm, um = st.mean(gc_list), st.mean(gu_list)
        w = "CONSTRAINED" if cm < um else "unrestricted"
        rows.append({"instance": name, "constrained_mean": cm, "unrestricted_mean": um,
                     "constrained": gc_list, "unrestricted": gu_list})
        gapk = "gap" if inst.optimal_makespan else "rel"
        print(f"{name:6}{cm:>12.1f}{gapk:>4}{um:>12.1f}{gapk:>4}  {w}")
        json.dump(rows, open(args.out, "w"), indent=2)
    nC = sum(1 for r in rows if r["constrained_mean"] < r["unrestricted_mean"])
    print(f"\nCONSTRAINED (ontology) transfers better on {nC}/{len(rows)} JSSP instances")


if __name__ == "__main__":
    main()
