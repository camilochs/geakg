#!/usr/bin/env python3
"""Run L0 pipeline: Pool + MetaGraph + ACO completo (sin synthesis synthesis).

Basado en exp_pure_mode_tsp.py pero usando L0 operator pool.
Incluye todas las funcionalidades del ACO original:
- Multi-instance con Instance Hardness Sampling
- MMAS con feromonas de operadores
- Incompatibility tracking
- Pruning de operadores
- Champion updates

Usage:
    # Single instance con metagraph predefinido
    uv run python scripts/run_l0_pipeline.py \
        --pool pools/tsp_pool.json \
        --instance data/instances/tsp/berlin52.tsp

    # Multi-instance con LLM metagraph
    uv run python scripts/run_l0_pipeline.py \
        --pool pools/tsp_pool.json \
        --instances data/instances/tsp/berlin52.tsp,data/instances/tsp/kroA100.tsp \
        --use-llm-metagraph \
        --model gpt-4o-mini

    # Multi-instance desde directorio
    uv run python scripts/run_l0_pipeline.py \
        --pool pools/tsp_pool.json \
        --instances-dir data/instances/tsp \
        --n-instances 5 \
        --use-llm-metagraph
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run L0 pipeline with full ACO")

    # Pool and model
    parser.add_argument("--pool", required=True, help="Path to L0 operator pool JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for metagraph")

    # Instance selection (single or multi)
    parser.add_argument("--instance", help="Path to single TSP instance")
    parser.add_argument("--instances", help="Comma-separated instance files")
    parser.add_argument("--instances-dir", help="Directory with TSP instances")
    parser.add_argument("--n-instances", type=int, help="Number of instances from dir")

    # ACO parameters
    parser.add_argument("--n-ants", type=int, default=15, help="Number of ants")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hardness-update-freq", type=int, default=10, help="Champion update frequency")

    # Options
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--output-dir", default="experiments/l0/results", help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--use-llm-metagraph",
        action="store_true",
        help="Use LLM to generate L2 (edges/weights) and L3 (conditions). Default: use predefined metagraph.",
    )

    args = parser.parse_args()

    # Validate instance arguments
    if not args.instance and not args.instances and not args.instances_dir:
        parser.error("Must specify --instance, --instances, or --instances-dir")

    # Logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    random.seed(args.seed)

    # Imports
    from src.geakg.offline.aco_trainer import MetaACOConfig, MetaACOSelector, OperatorMode
    from src.geakg.bindings import BindingRegistry
    from src.geakg.layers.l0.conditions import ExecutionContext
    from src.geakg.execution import evaluate_operator_path
    from src.geakg.instance_pool import InstancePool
    from src.geakg.layers.l1 import OperatorPool
    from src.geakg.layers.l1.synthesis_hook import L0SynthesisHook
    from src.geakg.visualization import visualize_akg
    from src.domains import get_domain_config

    # Setup session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / f"{timestamp}_l0"
    session_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(session_dir / "run.log", "w")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # =========================================================================
    # Load L0 Pool
    # =========================================================================
    log("=" * 60)
    log("L0 PIPELINE WITH FULL ACO")
    log("=" * 60)

    pool = OperatorPool.load(args.pool)
    log(f"Pool: {pool.total_operators} operators, {len(pool.roles)} roles")

    # =========================================================================
    # Load TSP Instances
    # =========================================================================
    log("")
    log("=" * 60)
    log("Loading TSP instances")
    log("=" * 60)

    instance_pool = InstancePool(domain="tsp", update_frequency=args.hardness_update_freq)

    if args.instance:
        # Single instance
        instance_pool.load_instances_from_files([args.instance])
    elif args.instances:
        # Multiple specific instances
        instance_files = [f.strip() for f in args.instances.split(",")]
        instance_pool.load_instances_from_files(instance_files)
    elif args.instances_dir:
        # Directory of instances
        instance_pool.load_instances_from_dir(args.instances_dir, limit=args.n_instances)

    if len(instance_pool) == 0:
        log("ERROR: No instances found")
        sys.exit(1)

    log(f"Loaded {len(instance_pool)} instances:")
    for inst in instance_pool.instances:
        opt_str = f" (opt={inst.optimal})" if inst.optimal else ""
        log(f"  - {inst.instance_id}: {inst.dimension} cities{opt_str}")

    domain_config = get_domain_config("tsp")
    avg_dimension = sum(inst.dimension for inst in instance_pool.instances) // len(instance_pool)

    # =========================================================================
    # Setup Pipeline with L0 Pool
    # =========================================================================
    log("")
    log("=" * 60)
    log("Setting up pipeline")
    log("=" * 60)

    # Reset bindings to pure mode
    BindingRegistry.reset()
    registry = BindingRegistry()
    bindings = registry.get_domain("tsp")

    # Create L0 synthesis hook with compiled operators
    synthesis_hook = L0SynthesisHook(pool)

    # Register L0 operators to bindings so ACO can select them
    synthesis_hook.register_operators_to_bindings(bindings)

    # Generate metagraph: either with LLM (L2/L3) or predefined
    if args.use_llm_metagraph:
        log("Using LLM to generate L2 (edges/weights) and L3 (conditions)")
        from src.geakg.layers.l0.topology_generator import L0MetaGraphGenerator
        from src.geakg.layers.l0.metagraph import InstantiatedGraph
        from src.llm.client import OpenAIClient

        llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=args.model,
        )
        metagraph_generator = L0MetaGraphGenerator(
            llm_client=llm_client,
            pool=pool,
            domain="TSP",
        )
        meta_graph = metagraph_generator.generate()
        if meta_graph is None:
            log("ERROR: Failed to generate metagraph with LLM")
            sys.exit(1)

        log(f"LLM-generated MetaGraph: {len(meta_graph.nodes)} roles, {len(meta_graph.edges)} edges")
        instantiated = InstantiatedGraph(meta_graph, bindings)

    else:
        log("Using predefined MetaGraph (no LLM for L2/L3)")
        from src.geakg.layers.l0.metagraph import InstantiatedGraph
        from src.geakg.layers.l0.patterns import create_hybrid_meta_graph

        meta_graph = create_hybrid_meta_graph()
        instantiated = InstantiatedGraph(meta_graph, bindings)

    # Create ACO selector with DYNAMIC mode for operator pheromones
    aco_config = MetaACOConfig(
        n_ants=args.n_ants,
        operator_mode=OperatorMode.DYNAMIC,  # Use operator pheromones
        enable_synthesis=False,  # No synthesis synthesis
        enable_conditions=True,
        enable_incompatibility_tracking=True,
        use_dynamic_energy=True,
    )
    selector = MetaACOSelector(instantiated, aco_config, synthesis_hook=synthesis_hook)

    log(f"MetaGraph: {len(meta_graph.nodes)} roles, {len(meta_graph.edges)} edges")
    log(f"Pool operators registered: {pool.total_operators}")
    log(f"ACO mode: DYNAMIC (operator pheromones enabled)")

    # Visualize initial AKG
    visualize_akg(
        selector.get_operator_pheromones(),
        str(session_dir / "akg_BEFORE.png"),
        title="L0 AKG - Initial",
        meta_graph=meta_graph,
    )

    # =========================================================================
    # Run ACO
    # =========================================================================
    log("")
    log("=" * 60)
    log("Running ACO optimization")
    log("=" * 60)

    # Track metrics
    best_weighted_gap = float("inf")
    best_operator_path = []
    best_role_path = []
    stagnation = 0
    iterations = 0
    convergence = []
    operator_usage = {}
    l0_operator_appearances = {}

    start_time = time.time()
    last_heartbeat = 0.0

    def evaluate_path(operator_path, instance_data):
        return evaluate_operator_path(operator_path, instance_data, domain_config, synthesis_hook)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= args.timeout:
            break

        iterations += 1

        # Heartbeat every 10 seconds
        if elapsed - last_heartbeat >= 10:
            last_heartbeat = elapsed
            incompat_stats = selector.get_incompatibility_stats()
            incompat_str = f" Incompat={incompat_stats.get('incompatible_count', 0)}" if incompat_stats else ""
            n_l0_used = len(set(k[1] for k in operator_usage.keys() if k[1] in [op.name for role in pool.roles for op in pool.get_operators_for_role(role)]))
            log(f"\033[90m[{elapsed:5.1f}s] Iter={iterations} Gap={best_weighted_gap:.2f}% L0used={n_l0_used} Stag={stagnation}{incompat_str}\033[0m")

        # Update champion periodically (recalculates hardness probabilities)
        if best_operator_path and instance_pool.should_update_champion(iterations):
            evals = instance_pool.update_champion(best_operator_path, evaluate_path, iterations)
            if evals > 0:
                # Log updated probabilities
                probs = {inst.instance_id: f"{inst.selection_probability:.2f}" for inst in instance_pool.instances}
                gaps = {inst.instance_id: f"{inst.champion_gap:.1f}%" for inst in instance_pool.instances}
                log(f"\033[90m  [CHAMPION] Updated - Gaps: {gaps}, Probs: {probs}\033[0m")

        # Construct solutions
        iteration_best_gap = float("inf")
        iteration_best_ant = None
        iteration_best_instance = None
        role_paths = []

        for _ in range(args.n_ants):
            # Sample instance based on hardness
            instance_info = instance_pool.sample_instance()

            # Construct solution
            ant = selector.construct_solution(problem_size=instance_info.dimension)

            if not ant.operator_path:
                continue

            # Evaluate
            fitness = evaluate_path(ant.operator_path, instance_info.instance_data)
            ant.fitness = fitness

            # Calculate gap
            if instance_info.optimal and instance_info.optimal > 0:
                ant_gap = 100 * (fitness - instance_info.optimal) / instance_info.optimal
            else:
                ant_gap = fitness / instance_info.dimension

            ant.gap = ant_gap
            role_paths.append(ant.role_path)

            # Track operator usage
            for role, op in zip(ant.role_path, ant.operator_path):
                operator_usage[(role, op)] = operator_usage.get((role, op), 0) + 1

            if ant_gap < iteration_best_gap:
                iteration_best_gap = ant_gap
                iteration_best_ant = ant
                iteration_best_instance = instance_info

        # Record operator results for pruning
        if iteration_best_ant:
            for op in iteration_best_ant.operator_path:
                selector.record_operator_result(
                    operator_id=op,
                    is_synth=False,  # L0 operators are not synthesis
                    fitness_before=best_weighted_gap,
                    fitness_after=iteration_best_gap,
                )

        # Record path outcomes for incompatibility tracking
        # Only record as failure if gap is significantly worse than a threshold
        # (not just "didn't improve global best" which would mark almost everything as failure)
        if iteration_best_ant:
            # Consider a path "failed" only if gap > 20% (clearly bad)
            # or if it's much worse than the current best (> 2x the gap)
            failure_threshold = max(20.0, best_weighted_gap * 2) if best_weighted_gap < float("inf") else 50.0
            is_failure = iteration_best_gap > failure_threshold
            selector.record_path_outcome(iteration_best_ant.role_path, is_failure=is_failure)

        # Update context
        context = ExecutionContext(
            generations_without_improvement=stagnation,
            population_diversity=len(set(tuple(p) for p in role_paths)) / max(len(role_paths), 1),
            current_fitness=iteration_best_gap if iteration_best_ant else float("inf"),
            best_fitness=best_weighted_gap,
        )
        selector.set_execution_context(context)

        # Update best - always evaluate on ALL instances for multi-instance mode
        if iteration_best_ant:
            candidate_path = iteration_best_ant.operator_path
            candidate_role_path = iteration_best_ant.role_path
            candidate_instance = iteration_best_instance.instance_id

            # Evaluate on ALL instances (multi-instance validation)
            if len(instance_pool) > 1:
                candidate_gaps = []
                per_instance_gaps = {}
                for inst in instance_pool.instances:
                    fitness = evaluate_path(candidate_path, inst.instance_data)
                    if inst.optimal and inst.optimal > 0:
                        gap = 100 * (fitness - inst.optimal) / inst.optimal
                    else:
                        gap = fitness / inst.dimension
                    candidate_gaps.append(gap)
                    per_instance_gaps[inst.instance_id] = gap
                candidate_avg_gap = sum(candidate_gaps) / len(candidate_gaps)
            else:
                candidate_avg_gap = iteration_best_gap
                per_instance_gaps = {candidate_instance: iteration_best_gap}

            if candidate_avg_gap < best_weighted_gap:
                best_weighted_gap = candidate_avg_gap
                best_operator_path = candidate_path.copy()
                best_role_path = candidate_role_path.copy()
                stagnation = 0
                convergence.append((elapsed, best_weighted_gap))

                selector.record_successful_path(candidate_role_path, candidate_path)

                # Track L0 operator appearances
                for role, op in zip(candidate_role_path, candidate_path):
                    if op not in l0_operator_appearances:
                        l0_operator_appearances[op] = (elapsed, best_weighted_gap, role)

                path_len = len(candidate_path)
                log(f"\033[92m+\033[0m [{elapsed:5.1f}s] AvgGap={best_weighted_gap:.2f}% len={path_len}")
                # Show per-instance gaps
                gaps_str = ", ".join(f"{k}:{v:.1f}%" for k, v in sorted(per_instance_gaps.items()))
                log(f"    Per-instance: {gaps_str}")
                log(f"    Operators: {candidate_path}")
            else:
                stagnation += 1
        else:
            stagnation += 1

        # Update pheromones (both role and operator level)
        if iteration_best_ant:
            selector.update_pheromones_for_path(
                iteration_best_ant.role_path,
                iteration_best_gap,
                operator_path=iteration_best_ant.operator_path,
            )
            selector.update_operator_pheromones(iteration_best_ant, iteration_best_gap)

        # Pruning check
        pruned = selector.check_and_prune_operators(iterations)
        if pruned:
            log(f"  [PRUNING] Removed {len(pruned)} operators: {pruned}")

    elapsed = time.time() - start_time

    # Final champion update
    if best_operator_path:
        instance_pool.update_champion(best_operator_path, evaluate_path, iterations, force=True)

    pool_stats = instance_pool.get_stats()

    # =========================================================================
    # Results
    # =========================================================================
    log("")
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)

    log(f"Best avg gap: {pool_stats['avg_champion_gap']:.2f}%")
    log(f"Time: {elapsed:.1f}s")
    log(f"Iterations: {iterations}")
    log(f"Best path: {best_role_path}")
    log(f"Best operators: {best_operator_path}")

    log(f"\n--- Per-Instance Gaps ---")
    for inst_id, gap in sorted(pool_stats.get("per_instance_gaps", {}).items(), key=lambda x: -x[1]):
        log(f"  {inst_id}: {gap:.2f}%")

    # Log Instance Hardness Sampling stats
    if len(instance_pool) > 1:
        log(f"\n--- Instance Hardness Sampling ---")
        log(f"  Hardest: {pool_stats.get('hardest_instance', 'N/A')}")
        log(f"  Easiest: {pool_stats.get('easiest_instance', 'N/A')}")
        log(f"  Selection counts:")
        for inst_id, count in sorted(pool_stats.get("selection_counts", {}).items(), key=lambda x: -x[1]):
            prob = pool_stats.get("probabilities", {}).get(inst_id, 0)
            log(f"    {inst_id}: {count} times (prob={prob:.2f})")

    # Log IncompatibilityTracker results
    incompat_stats = selector.get_incompatibility_stats()
    if incompat_stats:
        log(f"\n--- IncompatibilityTracker ---")
        log(f"  Failures tracked: {incompat_stats.get('total_failures', 0)}")
        log(f"  Successes tracked: {incompat_stats.get('total_successes', 0)}")
        log(f"  Incompatible transitions: {incompat_stats.get('incompatible_count', 0)}")
        incompatible = selector.get_incompatible_transitions()
        if incompatible:
            log("  Penalized transitions:")
            for src, tgt in sorted(incompatible)[:10]:  # Top 10
                log(f"    {src} → {tgt}")

    # Log operator usage
    log(f"\n--- Operator Usage (top 15) ---")
    sorted_usage = sorted(operator_usage.items(), key=lambda x: -x[1])[:15]
    for (role, op), count in sorted_usage:
        log(f"  {op} ({role}): {count}")

    # Visualize final AKG
    visualize_akg(
        selector.get_operator_pheromones(),
        str(session_dir / "akg_AFTER.png"),
        title=f"L0 AKG - Final (Gap={pool_stats['avg_champion_gap']:.2f}%)",
        meta_graph=meta_graph,
    )

    # Save results
    output_data = {
        "aggregate_gap": pool_stats['avg_champion_gap'],
        "per_instance_gaps": pool_stats.get("per_instance_gaps", {}),
        "elapsed_seconds": elapsed,
        "iterations": iterations,
        "seed": args.seed,
        "pool_path": args.pool,
        "pool_operators": pool.total_operators,
        "best_role_path": best_role_path,
        "best_operator_path": best_operator_path,
        "convergence": convergence,
        "operator_usage": {f"{r}:{o}": c for (r, o), c in sorted_usage},
        "l0_operator_appearances": {
            op: {"time": t, "gap": g, "role": r}
            for op, (t, g, r) in l0_operator_appearances.items()
        },
        "incompatibility_stats": incompat_stats,
        "n_instances": len(instance_pool),
        "n_ants": args.n_ants,
        "use_llm_metagraph": args.use_llm_metagraph,
    }

    results_file = args.output or str(session_dir / "results.json")
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    log(f"\nSaved: {results_file}")

    log_file.close()

    print(f"\n{'='*60}")
    print(f"L0 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Instances: {len(instance_pool)}")
    print(f"Best avg gap: {pool_stats['avg_champion_gap']:.2f}%")
    print(f"Time: {elapsed:.1f}s | Iterations: {iterations}")
    print(f"Session: {session_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
