#!/usr/bin/env python3
"""Run NAS-Bench-201 experiments for the GEAKG paper.

Experiments:
  1. GEAKG-NAS on CIFAR-10 (vs baselines)
  2. GEAKG-NAS on CIFAR-100
  3. Transfer CIFAR-10 -> CIFAR-100
  4. Transfer CIFAR-10 -> ImageNet-16-120
  5. Pheromone analysis (qualitative)
  6. LLM sweep for L1 operator quality

Usage:
    python scripts/run_nas_benchmark.py --experiment direct --dataset cifar10
    python scripts/run_nas_benchmark.py --experiment transfer
    python scripts/run_nas_benchmark.py --experiment llm-sweep --llm gpt5.2
    python scripts/run_nas_benchmark.py --experiment all --n-runs 5
    python scripts/run_nas_benchmark.py --quick  # Smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.geakg.safe_exec import safe_call_operator

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats

from src.domains.nas.cell_architecture import (
    CellArchitecture,
    NUM_EDGES,
    NUM_OPS,
)
from src.domains.nas.nasbench_evaluator import NASBench201Evaluator


# =============================================================================
# BASELINES
# =============================================================================


def random_search(
    evaluator: NASBench201Evaluator,
    n_evals: int,
    dataset: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Random Search baseline: sample uniformly from the search space."""
    rng = random.Random(seed)
    best_acc = -1.0
    best_arch = None
    convergence = []
    t0 = time.time()

    for _ in range(n_evals):
        arch = CellArchitecture.random(rng)
        acc = evaluator.evaluate(arch, dataset)
        if acc > best_acc:
            best_acc = acc
            best_arch = arch
        convergence.append(best_acc)

    return {
        "method": "random_search",
        "best_accuracy": best_acc,
        "best_architecture": best_arch.to_dict() if best_arch else None,
        "convergence": convergence,
        "n_evals": n_evals,
        "wall_time_s": time.time() - t0,
    }


def regularized_evolution(
    evaluator: NASBench201Evaluator,
    n_evals: int,
    dataset: str,
    population_size: int = 50,
    tournament_size: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Regularized Evolution baseline (Real et al. 2019).

    Age-regularized evolution with tournament selection and single-edge mutation.
    """
    rng = random.Random(seed)
    population: list[tuple[CellArchitecture, float]] = []
    best_acc = -1.0
    best_arch = None
    convergence = []
    t0 = time.time()

    # Initialize population
    for _ in range(min(population_size, n_evals)):
        arch = CellArchitecture.random(rng)
        acc = evaluator.evaluate(arch, dataset)
        population.append((arch, acc))
        if acc > best_acc:
            best_acc = acc
            best_arch = arch
        convergence.append(best_acc)

    remaining = n_evals - len(population)

    for _ in range(remaining):
        # Tournament selection
        sample = rng.sample(population, min(tournament_size, len(population)))
        parent = max(sample, key=lambda x: x[1])[0]

        # Mutation: change 1 random edge
        child = parent.copy()
        edge_idx = rng.randint(0, NUM_EDGES - 1)
        child.edges[edge_idx] = rng.randint(0, NUM_OPS - 1)

        acc = evaluator.evaluate(child, dataset)
        population.append((child, acc))

        if acc > best_acc:
            best_acc = acc
            best_arch = child

        # Kill oldest (age regularization)
        if len(population) > population_size:
            population.pop(0)

        convergence.append(best_acc)

    return {
        "method": "regularized_evolution",
        "best_accuracy": best_acc,
        "best_architecture": best_arch.to_dict() if best_arch else None,
        "convergence": convergence,
        "n_evals": n_evals,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "wall_time_s": time.time() - t0,
    }


def bayesian_optimization(
    evaluator: NASBench201Evaluator,
    n_evals: int,
    dataset: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Bayesian Optimization baseline (GP + Expected Improvement).

    Uses scikit-optimize's gp_minimize over the 6-integer NAS-Bench-201
    search space (each edge in [0, NUM_OPS-1]).
    """
    from skopt import gp_minimize
    from skopt.space import Integer

    dimensions = [Integer(0, NUM_OPS - 1, name=f"edge_{i}") for i in range(NUM_EDGES)]

    best_acc = -1.0
    best_arch = None
    convergence: list[float] = []
    t0 = time.time()

    def objective(params: list[int]) -> float:
        nonlocal best_acc, best_arch
        arch = CellArchitecture(edges=list(params))
        acc = evaluator.evaluate(arch, dataset)
        if acc > best_acc:
            best_acc = acc
            best_arch = arch
        convergence.append(best_acc)
        return -acc  # gp_minimize minimizes

    n_initial = min(20, n_evals // 3)
    gp_minimize(
        objective,
        dimensions,
        n_calls=n_evals,
        n_initial_points=n_initial,
        acq_func="EI",
        random_state=seed,
    )

    return {
        "method": "bayesian_optimization",
        "best_accuracy": best_acc,
        "best_architecture": best_arch.to_dict() if best_arch else None,
        "convergence": convergence,
        "n_evals": n_evals,
        "wall_time_s": time.time() - t0,
    }


def bayesian_optimization_timed(
    evaluator: NASBench201Evaluator,
    n_evals: int,
    dataset: str,
    wall_time_budget_s: float,
    seed: int = 42,
) -> dict[str, Any]:
    """BO with wall-clock time limit (fair comparison with GEAKG).

    Uses scikit-optimize's Optimizer ask/tell loop so we can check the
    clock between evaluations and stop when the budget is exhausted.
    """
    from skopt import Optimizer
    from skopt.space import Integer

    dimensions = [Integer(0, NUM_OPS - 1, name=f"edge_{i}") for i in range(NUM_EDGES)]
    opt = Optimizer(
        dimensions, base_estimator="GP", acq_func="EI", random_state=seed,
    )

    best_acc = -1.0
    best_arch = None
    convergence: list[float] = []
    eval_count = 0
    t0 = time.time()

    for _ in range(n_evals):
        if time.time() - t0 > wall_time_budget_s:
            break
        x = opt.ask()
        arch = CellArchitecture(edges=list(x))
        acc = evaluator.evaluate(arch, dataset)
        opt.tell(x, -acc)
        eval_count += 1
        if acc > best_acc:
            best_acc = acc
            best_arch = arch
        convergence.append(best_acc)

    wall_time = time.time() - t0

    return {
        "method": "bo_timed",
        "best_accuracy": best_acc,
        "best_architecture": best_arch.to_dict() if best_arch else None,
        "convergence": convergence,
        "n_evals": eval_count,
        "wall_time_s": wall_time,
        "wall_time_budget_s": wall_time_budget_s,
    }


# =============================================================================
# GEAKG-NAS
# =============================================================================


def geakg_nas(
    evaluator: NASBench201Evaluator,
    dataset: str,
    n_ants: int = 10,
    n_iterations: int = 50,
    seed: int = 42,
    n_evals_budget: int = 500,
    initial_pheromones: dict[tuple[str, str], float] | None = None,
    operator_pool: Any | None = None,
) -> dict[str, Any]:
    """GEAKG-NAS: Navigate the procedural KG to design cell architectures.

    Uses NASRoleSchema (18 roles) + MetaGraph + ACO pheromones.
    Each ant starts with a random CellArchitecture, traverses the MetaGraph,
    applies operators at each role, and evaluates the result.

    Budget is controlled by n_evals_budget (total evaluations), NOT
    n_ants * n_iterations. ACO stops when budget is exhausted.

    Args:
        evaluator: NAS-Bench-201 evaluator.
        dataset: Dataset name.
        n_ants: Number of ants per iteration.
        n_iterations: Maximum ACO iterations.
        seed: Random seed.
        n_evals_budget: Total evaluation budget (same as baselines).
        initial_pheromones: Optional warm-start pheromones from transfer.
        operator_pool: Optional OperatorPool from L1 synthesis. When provided,
                      compiled L1 operators replace base A₀ operators.

    Returns:
        Results dict with accuracy, convergence, pheromone data, and stats.
    """
    from src.geakg.core.schemas.nas import NASRoleSchema
    from src.geakg.layers.l0.patterns import create_nas_meta_graph
    from src.geakg.generic_operators.cell_architecture import (
        CELL_ARCHITECTURE_OPERATORS,
    )

    rng = random.Random(seed)
    random.seed(seed)
    t0 = time.time()

    schema = NASRoleSchema()
    mg = create_nas_meta_graph(schema)

    # Operator dispatch: role -> list of callable functions
    # If operator_pool is provided (L1 synthesis), compile and use those.
    # Otherwise fall back to base A₀ operators.
    operators_by_role: dict[str, list] = {}
    l1_operator_mode = "base_A0"

    if operator_pool is not None:
        l1_operator_mode = "l1_synthesized"
        compiled_pool = _compile_operator_pool(operator_pool)
        for role in schema.get_all_roles():
            funcs = compiled_pool.get(role, [])
            if funcs:
                operators_by_role[role] = funcs
            else:
                # Fallback to base operators for roles without L1 variants
                ops = CELL_ARCHITECTURE_OPERATORS.get_operators(role)
                if ops:
                    operators_by_role[role] = ops
        logger.info(
            f"[GEAKG-NAS] Using L1 operators: "
            f"{sum(len(v) for v in compiled_pool.values())} compiled from pool"
        )
    else:
        for role in schema.get_all_roles():
            ops = CELL_ARCHITECTURE_OPERATORS.get_operators(role)
            if ops:
                operators_by_role[role] = ops

    # Pheromone matrix for role transitions
    pheromones: dict[tuple[str, str], float] = {}
    for (src, tgt), edge in mg.edges.items():
        pheromones[(src, tgt)] = max(0.01, edge.weight)

    # Warm-start: overlay source pheromones if provided
    if initial_pheromones is not None:
        transferred = 0
        for key, tau in initial_pheromones.items():
            if key in pheromones:
                pheromones[key] = tau
                transferred += 1
        logger.info(f"[Transfer] Warm-started {transferred}/{len(pheromones)} pheromone edges")

    # ACO parameters
    alpha = 2.0
    beta = 2.0
    rho = 0.1
    tau_min = 0.001
    tau_max = 1.0

    best_acc = -1.0
    best_arch = None
    convergence = []  # Best accuracy per evaluation (for fair convergence comparison)
    eval_count = 0
    path_counter: Counter = Counter()
    role_frequency: Counter = Counter()
    operator_failures = 0
    stagnation_counter = 0
    last_best_acc = -1.0

    # Dummy context for operators
    class _DummyCtx:
        def evaluate(self, sol):
            return evaluator.evaluate(sol, dataset)

    ctx = _DummyCtx()

    for iteration in range(n_iterations):
        if eval_count >= n_evals_budget:
            break

        iteration_best_acc = -1.0
        iteration_best_path: list[str] = []

        for _ in range(n_ants):
            if eval_count >= n_evals_budget:
                break

            # Start with random cell
            arch = CellArchitecture.random(rng)

            # Select entry role (topology)
            entry_roles = list(schema.get_roles_by_category("topology"))
            current_role = _select_role(entry_roles, None, pheromones, alpha, beta, rng)

            path = [current_role]
            energy = rng.uniform(4.0, 10.0)

            # Traverse MetaGraph
            while energy > 0:
                # Apply operator at current role
                if current_role in operators_by_role:
                    ops = operators_by_role[current_role]
                    op = rng.choice(ops)
                    new_arch = safe_call_operator(op, arch, ctx)
                    if new_arch is not None and (
                        hasattr(new_arch, "edges")
                        and len(new_arch.edges) == NUM_EDGES
                        and all(0 <= e < NUM_OPS for e in new_arch.edges)
                    ):
                        arch = new_arch
                    else:
                        operator_failures += 1

                energy -= 1.0
                role_frequency[current_role] += 1

                # Select next role
                successors = mg.get_successors(current_role)
                if not successors:
                    break

                next_role = _select_role(
                    successors, current_role, pheromones, alpha, beta, rng
                )
                path.append(next_role)
                current_role = next_role

            # Evaluate final architecture
            acc = evaluator.evaluate(arch, dataset)
            eval_count += 1

            if acc > iteration_best_acc:
                iteration_best_acc = acc
                iteration_best_path = path

            if acc > best_acc:
                best_acc = acc
                best_arch = arch

            convergence.append(best_acc)
            path_counter[tuple(path)] += 1

        # Pheromone update (MMAS)
        # Evaporation
        for key in pheromones:
            pheromones[key] *= (1 - rho)

        # Deposit on iteration-best path
        # Deposit proportional to accuracy (higher accuracy = more deposit)
        if len(iteration_best_path) >= 2 and iteration_best_acc > 0:
            deposit = iteration_best_acc / 100.0  # Normalized [0, 1]
            for i in range(len(iteration_best_path) - 1):
                key = (iteration_best_path[i], iteration_best_path[i + 1])
                if key in pheromones:
                    pheromones[key] += deposit

        # Stagnation detection
        if best_acc > last_best_acc:
            stagnation_counter = 0
            last_best_acc = best_acc
        else:
            stagnation_counter += 1

        # Reset pheromones if stagnated (MMAS)
        if stagnation_counter >= 20:
            for key in pheromones:
                pheromones[key] = tau_max
            stagnation_counter = 0

        # Clamp pheromones
        for key in pheromones:
            pheromones[key] = max(tau_min, min(tau_max, pheromones[key]))

        if (iteration + 1) % 10 == 0:
            logger.info(
                f"  Iteration {iteration + 1}/{n_iterations}: "
                f"best_acc={best_acc:.2f}%, evals={eval_count}/{n_evals_budget}"
            )

    wall_time = time.time() - t0

    # Top-5 paths
    top_paths = path_counter.most_common(5)

    # Return pheromones as tuple-keyed dict (for transfer reuse)
    pheromones_raw = dict(pheromones)

    return {
        "method": "geakg_nas",
        "best_accuracy": best_acc,
        "best_architecture": best_arch.to_dict() if best_arch else None,
        "convergence": convergence,
        "n_evals": eval_count,
        "n_ants": n_ants,
        "n_iterations": n_iterations,
        "n_evals_budget": n_evals_budget,
        "wall_time_s": wall_time,
        "operator_failures": operator_failures,
        "pheromones_display": {f"{k[0]}->{k[1]}": v for k, v in pheromones.items()},
        "pheromones_raw": pheromones_raw,
        "top_paths": [
            {"path": list(p), "count": c} for p, c in top_paths
        ],
        "role_frequency": dict(role_frequency),
        "was_transferred": initial_pheromones is not None,
        "operator_mode": l1_operator_mode,
    }


def _compile_operator_pool(pool: Any) -> dict[str, list]:
    """Compile an OperatorPool's code strings into callable functions.

    Args:
        pool: OperatorPool from L1 synthesis.

    Returns:
        Dict mapping role -> list of callable functions.
    """
    compiled: dict[str, list] = {}

    for role in pool.roles:
        funcs = []
        for op in pool.get_operators_for_role(role):
            try:
                namespace: dict = {}
                exec(compile(op.code, f"<{op.name}>", "exec"), namespace)
                # Find the callable
                fn = None
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith("_"):
                        fn = obj
                        break
                if fn is not None:
                    funcs.append(fn)
            except Exception as e:
                logger.debug(f"[L1] Failed to compile {op.name}: {e}")

        if funcs:
            compiled[role] = funcs

    return compiled


def _select_role(
    candidates: list[str],
    current: str | None,
    pheromones: dict[tuple[str, str], float],
    alpha: float,
    beta: float,
    rng: random.Random | None = None,
) -> str:
    """ACO probabilistic role selection."""
    r = rng or random
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Exploration with 10% probability
    if r.random() < 0.1:
        return r.choice(candidates)

    probs = []
    for c in candidates:
        tau = pheromones.get((current, c), 0.5) if current else 1.0
        eta = 1.0
        probs.append((tau ** alpha) * (eta ** beta))

    total = sum(probs)
    if total == 0:
        return r.choice(candidates)

    probs = [p / total for p in probs]
    val = r.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if val <= cumulative:
            return candidates[i]
    return candidates[-1]


# =============================================================================
# EXPERIMENTS
# =============================================================================


def experiment_direct(
    dataset: str,
    n_runs: int,
    n_ants: int,
    n_iterations: int,
    n_evals: int,
    seed: int,
    evaluator: NASBench201Evaluator,
) -> dict[str, Any]:
    """Experiment 1/2: GEAKG-NAS vs baselines on a single dataset.

    All methods use the same evaluation budget (n_evals).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: Direct search on {dataset} (budget={n_evals} evals)")
    logger.info(f"{'='*60}")

    all_results: dict[str, list] = defaultdict(list)

    for run in range(n_runs):
        run_seed = seed + run
        logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={run_seed}) ---")

        # GEAKG-NAS (same budget as baselines)
        logger.info("[GEAKG-NAS] Running...")
        res = geakg_nas(
            evaluator, dataset, n_ants, n_iterations, run_seed,
            n_evals_budget=n_evals,
        )
        all_results["geakg_nas"].append(res)
        logger.info(
            f"[GEAKG-NAS] Best: {res['best_accuracy']:.2f}% "
            f"({res['n_evals']} evals, {res['wall_time_s']:.1f}s, "
            f"{res['operator_failures']} op failures)"
        )

        # Random Search
        logger.info("[Random Search] Running...")
        res = random_search(evaluator, n_evals, dataset, run_seed)
        all_results["random_search"].append(res)
        logger.info(f"[Random Search] Best: {res['best_accuracy']:.2f}%")

        # Regularized Evolution
        logger.info("[Reg. Evolution] Running...")
        res = regularized_evolution(evaluator, n_evals, dataset, seed=run_seed)
        all_results["reg_evolution"].append(res)
        logger.info(f"[Reg. Evolution] Best: {res['best_accuracy']:.2f}%")

        # Bayesian Optimization (unlimited)
        logger.info("[Bayesian Opt.] Running...")
        res = bayesian_optimization(evaluator, n_evals, dataset, run_seed)
        all_results["bayesian_opt"].append(res)
        logger.info(f"[Bayesian Opt.] Best: {res['best_accuracy']:.2f}%")

        # Bayesian Optimization (time-matched, 2s budget)
        logger.info("[BO timed 2s] Running...")
        res = bayesian_optimization_timed(
            evaluator, n_evals, dataset, wall_time_budget_s=2.0, seed=run_seed,
        )
        all_results["bo_timed_2s"].append(res)
        logger.info(
            f"[BO timed 2s] Best: {res['best_accuracy']:.2f}% "
            f"({res['n_evals']} evals in {res['wall_time_s']:.2f}s)"
        )

    summary = _aggregate_results(all_results, dataset)
    return summary


def experiment_transfer(
    source_dataset: str,
    target_dataset: str,
    n_runs: int,
    n_ants: int,
    n_iterations: int,
    n_evals: int,
    seed: int,
    evaluator: NASBench201Evaluator,
) -> dict[str, Any]:
    """Experiment 3/4: Transfer from source to target dataset.

    Pipeline:
    1. Train GEAKG on source (n_evals budget)
    2. Zero-shot: evaluate source's best architecture on target
    3. Transfer + fine-tune: warm-start pheromones on target (half budget)
    4. Cold start: run GEAKG on target from scratch (same total budget)
    5. Random search baseline on target (same budget)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: Transfer {source_dataset} -> {target_dataset}")
    logger.info(f"{'='*60}")

    all_results: dict[str, list] = defaultdict(list)

    for run in range(n_runs):
        run_seed = seed + run
        logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={run_seed}) ---")

        # Step 1: Train GEAKG on source dataset
        logger.info(f"[Transfer] Training on {source_dataset} ({n_evals} evals)...")
        source_res = geakg_nas(
            evaluator, source_dataset, n_ants, n_iterations, run_seed,
            n_evals_budget=n_evals,
        )
        source_pheromones = source_res.get("pheromones_raw", {})
        logger.info(f"[Transfer] Source best: {source_res['best_accuracy']:.2f}%")

        # Step 2: Zero-shot (evaluate source best arch on target, 0 extra evals)
        logger.info(f"[Transfer] Zero-shot on {target_dataset}...")
        if source_res["best_architecture"]:
            best_source_arch = CellArchitecture.from_dict(source_res["best_architecture"])
            zero_shot_acc = evaluator.evaluate(best_source_arch, target_dataset)
        else:
            zero_shot_acc = 0.0
        all_results["transfer_zero_shot"].append({
            "method": "transfer_zero_shot",
            "best_accuracy": zero_shot_acc,
            "source_accuracy": source_res["best_accuracy"],
            "n_evals_source": source_res["n_evals"],
            "n_evals_target": 1,
        })
        logger.info(f"[Transfer] Zero-shot: {zero_shot_acc:.2f}%")

        # Step 3: Transfer + fine-tune (warm-start with source pheromones)
        finetune_budget = n_evals // 2
        logger.info(
            f"[Transfer] Fine-tuning on {target_dataset} "
            f"with warm-start pheromones ({finetune_budget} evals)..."
        )
        finetune_res = geakg_nas(
            evaluator, target_dataset, n_ants, n_iterations,
            run_seed + 1000,
            n_evals_budget=finetune_budget,
            initial_pheromones=source_pheromones,  # WARM-START
        )
        all_results["transfer_finetune"].append({
            "method": "transfer_finetune",
            "best_accuracy": finetune_res["best_accuracy"],
            "source_accuracy": source_res["best_accuracy"],
            "n_evals_source": source_res["n_evals"],
            "n_evals_target": finetune_res["n_evals"],
            "wall_time_s": finetune_res["wall_time_s"],
            "was_transferred": True,
        })
        logger.info(
            f"[Transfer] Fine-tuned: {finetune_res['best_accuracy']:.2f}% "
            f"({finetune_res['n_evals']} target evals)"
        )

        # Step 4: Cold start on target (same budget as source + finetune)
        cold_budget = n_evals
        logger.info(f"[Transfer] Cold start on {target_dataset} ({cold_budget} evals)...")
        cold_res = geakg_nas(
            evaluator, target_dataset, n_ants, n_iterations, run_seed,
            n_evals_budget=cold_budget,
        )
        all_results["cold_start"].append(cold_res)
        logger.info(f"[Transfer] Cold start: {cold_res['best_accuracy']:.2f}%")

        # Step 5: Random search on target
        logger.info(f"[Transfer] Random search on {target_dataset} ({n_evals} evals)...")
        rand_res = random_search(evaluator, n_evals, target_dataset, run_seed)
        all_results["random_search"].append(rand_res)
        logger.info(f"[Transfer] Random search: {rand_res['best_accuracy']:.2f}%")

    summary = _aggregate_results(all_results, target_dataset)
    summary["source_dataset"] = source_dataset
    summary["target_dataset"] = target_dataset
    return summary


def experiment_pheromone_analysis(
    dataset: str,
    n_ants: int,
    n_iterations: int,
    n_evals: int,
    seed: int,
    evaluator: NASBench201Evaluator,
) -> dict[str, Any]:
    """Experiment 5: Qualitative analysis of learned pheromones."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: Pheromone analysis on {dataset}")
    logger.info(f"{'='*60}")

    res = geakg_nas(evaluator, dataset, n_ants, n_iterations, seed, n_evals_budget=n_evals)

    pheromone_data = res.get("pheromones_display", {})
    sorted_phero = sorted(pheromone_data.items(), key=lambda x: x[1], reverse=True)

    logger.info("\nTop-10 pheromone transitions:")
    for edge, value in sorted_phero[:10]:
        logger.info(f"  {edge}: {value:.4f}")

    logger.info("\nBottom-10 pheromone transitions:")
    for edge, value in sorted_phero[-10:]:
        logger.info(f"  {edge}: {value:.4f}")

    logger.info("\nTop-5 most common paths:")
    for p in res.get("top_paths", [])[:5]:
        logger.info(f"  {' -> '.join(p['path'][:6])}... (count={p['count']})")

    logger.info("\nRole usage frequency:")
    role_freq = res.get("role_frequency", {})
    total_uses = sum(role_freq.values()) if role_freq else 1
    for role, count in sorted(role_freq.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / total_uses
        logger.info(f"  {role}: {count} ({pct:.1f}%)")

    # Role coverage
    from src.geakg.core.schemas.nas import NASRoleSchema
    all_roles = set(NASRoleSchema().get_all_roles())
    used_roles = set(role_freq.keys())
    unused = all_roles - used_roles
    if unused:
        logger.warning(f"\nUnused roles ({len(unused)}): {sorted(unused)}")
    else:
        logger.info(f"\nFull role coverage: all {len(all_roles)} roles used")

    # Symbolic rules
    rules = []
    for edge_str, value in sorted_phero[:10]:
        src, tgt = edge_str.split("->")
        rules.append({
            "rule": f"After {src}, prefer {tgt}",
            "confidence": round(value, 4),
        })

    return {
        "experiment": "pheromone_analysis",
        "dataset": dataset,
        "pheromones": pheromone_data,
        "top_paths": res.get("top_paths", []),
        "role_frequency": role_freq,
        "role_coverage": len(used_roles) / len(all_roles),
        "symbolic_rules": rules,
        "best_accuracy": res["best_accuracy"],
        "operator_failures": res.get("operator_failures", 0),
    }


def experiment_llm_sweep(
    dataset: str,
    llm: str,
    n_runs: int,
    n_ants: int,
    n_iterations: int,
    n_evals: int,
    seed: int,
    evaluator: NASBench201Evaluator,
    l1_token_budget: int = 15_000,
) -> dict[str, Any]:
    """Experiment 6: LLM sweep for L1 operator quality.

    For llm="none": runs with base operators A₀.
    For llm="gpt-4o-mini"/etc: runs L1 synthesis via NASGeneratorL1,
    then uses the generated operator pool in GEAKG-NAS.

    The L1 pool is generated ONCE per LLM (offline phase), then used
    across all runs (online phase). This mirrors the real GEAKG pipeline.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: LLM sweep (llm={llm}) on {dataset}")
    logger.info(f"{'='*60}")

    pool = None
    l1_stats: dict[str, Any] = {}
    t0_l1 = time.time()

    if llm != "none":
        # ---- L1 Synthesis Phase (offline, once per LLM) ----
        logger.info(f"[L1] Generating operator pool with {llm} ({l1_token_budget} tokens)...")
        try:
            from src.geakg.layers.l1.nas_generator import NASGeneratorL1, NASL1Config
            from src.llm.client import OpenAIClient

            llm_client = OpenAIClient(model=llm, temperature=0.7)
            config = NASL1Config(
                max_tokens=l1_token_budget,
                pool_size_per_role=3,
                temperature=0.7,
                seed=seed,
            )
            generator = NASGeneratorL1(llm_client, config=config)
            pool = generator.generate(evaluator)

            l1_stats = {
                "llm": llm,
                "tokens_used": generator.tokens_used,
                "total_operators": pool.total_operators,
                "roles_covered": len(pool.roles),
                "generation_stats": dict(generator.generation_stats),
                "l1_time_s": time.time() - t0_l1,
            }
            logger.info(
                f"[L1] Pool ready: {pool.total_operators} operators, "
                f"{generator.tokens_used} tokens, {time.time() - t0_l1:.1f}s"
            )

            # Save pool for reproducibility
            pool_path = Path("results/nas_bench/pools")
            pool_path.mkdir(parents=True, exist_ok=True)
            pool.save(pool_path / f"nas_pool_{llm.replace('/', '_')}_{seed}.json")

        except Exception as e:
            logger.error(f"[L1] Synthesis failed: {e}. Falling back to A₀.")
            pool = None
            l1_stats["error"] = str(e)

    # ---- ACO Phase (online, per run) ----
    all_results = []

    for run in range(n_runs):
        run_seed = seed + run
        logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={run_seed}) ---")

        res = geakg_nas(
            evaluator, dataset, n_ants, n_iterations, run_seed,
            n_evals_budget=n_evals,
            operator_pool=pool,
        )
        res["llm"] = llm
        all_results.append(res)

        logger.info(
            f"[LLM={llm}] Best: {res['best_accuracy']:.2f}% "
            f"(mode={res.get('operator_mode', 'unknown')})"
        )

    accuracies = [r["best_accuracy"] for r in all_results]
    return {
        "experiment": "llm_sweep",
        "dataset": dataset,
        "llm": llm,
        "n_runs": n_runs,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "best_accuracy": float(np.max(accuracies)),
        "operator_mode": "l1_synthesized" if pool else "base_A0",
        "l1_stats": l1_stats,
        "runs": all_results,
    }


# =============================================================================
# AGGREGATION & STATS
# =============================================================================


def _aggregate_results(
    all_results: dict[str, list],
    dataset: str,
) -> dict[str, Any]:
    """Aggregate results across runs with statistical tests."""
    summary: dict[str, Any] = {
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
        "methods": {},
    }

    for method, runs in all_results.items():
        accuracies = [r["best_accuracy"] for r in runs]
        wall_times = [r.get("wall_time_s", 0) for r in runs]

        method_stats: dict[str, Any] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "best_accuracy": float(np.max(accuracies)),
            "worst_accuracy": float(np.min(accuracies)),
            "n_runs": len(runs),
            "mean_wall_time_s": float(np.mean(wall_times)) if wall_times else 0,
        }

        # Convergence curves (averaged across runs if available)
        curves = [r.get("convergence", []) for r in runs if r.get("convergence")]
        if curves:
            min_len = min(len(c) for c in curves)
            if min_len > 0:
                trimmed = [c[:min_len] for c in curves]
                mean_curve = np.mean(trimmed, axis=0).tolist()
                std_curve = np.std(trimmed, axis=0).tolist()
                method_stats["mean_convergence"] = mean_curve
                method_stats["std_convergence"] = std_curve

        method_stats["runs"] = runs
        summary["methods"][method] = method_stats

    # Statistical tests: GEAKG vs each baseline
    geakg_accs = [r["best_accuracy"] for r in all_results.get("geakg_nas", [])]
    if len(geakg_accs) >= 2:
        for baseline in ["random_search", "reg_evolution", "bayesian_opt", "cold_start"]:
            baseline_accs = [r["best_accuracy"] for r in all_results.get(baseline, [])]
            if len(baseline_accs) >= 2 and len(geakg_accs) == len(baseline_accs):
                try:
                    stat, p_value = scipy_stats.wilcoxon(geakg_accs, baseline_accs)
                    if baseline in summary["methods"]:
                        summary["methods"][baseline]["vs_geakg_wilcoxon_p"] = float(p_value)
                        summary["methods"][baseline]["vs_geakg_significant"] = p_value < 0.05
                except Exception:
                    pass

    # Print table
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS SUMMARY ({dataset})")
    logger.info(f"{'='*60}")
    logger.info(f"{'Method':<25} {'Mean ± Std':>15} {'Best':>8} {'Time':>7}")
    logger.info("-" * 57)
    for method, stats in summary["methods"].items():
        mean = stats["mean_accuracy"]
        std = stats["std_accuracy"]
        best = stats["best_accuracy"]
        t = stats.get("mean_wall_time_s", 0)
        sig = ""
        if "vs_geakg_wilcoxon_p" in stats:
            p = stats["vs_geakg_wilcoxon_p"]
            sig = f"  p={p:.3f}"
        logger.info(f"{method:<25} {mean:>6.2f} ± {std:<6.2f} {best:>7.2f}% {t:>5.1f}s{sig}")

    return summary


def save_results(results: dict[str, Any], output_dir: str, experiment: str) -> Path:
    """Save results to JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nasbench_{experiment}_{timestamp}.json"
    filepath = out_path / filename

    # Remove non-serializable pheromones_raw from output
    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v) for k, v in obj.items()
                if k != "pheromones_raw"
            }
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    cleaned = _clean(results)

    with open(filepath, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NAS-Bench-201 experiments for GEAKG paper"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="direct",
        choices=["direct", "transfer", "pheromone", "llm-sweep", "all"],
        help="Experiment to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset (cifar10, cifar100, ImageNet16-120)",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default="cifar100",
        help="Target dataset for transfer experiment",
    )
    parser.add_argument("--llm", type=str, default="none", help="LLM for L1 synthesis")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--n-ants", type=int, default=10, help="Ants per iteration")
    parser.add_argument("--n-iterations", type=int, default=50, help="ACO iterations")
    parser.add_argument("--n-evals", type=int, default=500, help="Evaluation budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--nasbench-path",
        type=str,
        default=None,
        help="Path to NAS-Bench-201 data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/nas_bench",
        help="Output directory",
    )
    parser.add_argument(
        "--l1-tokens",
        type=int,
        default=15_000,
        help="Token budget for L1 synthesis (per LLM)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")

    args = parser.parse_args()

    if args.quick:
        args.n_runs = 2
        args.n_ants = 5
        args.n_iterations = 20
        args.n_evals = 50

    # Load API key from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Create evaluator
    evaluator = NASBench201Evaluator(
        nasbench_path=args.nasbench_path,
        dataset=args.dataset,
        seed=args.seed,
    )

    logger.info(f"Evaluator: dataset={evaluator.dataset}")
    logger.info(f"Budget: {args.n_evals} evals, {args.n_runs} runs, seed={args.seed}")

    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = ["direct", "transfer", "pheromone", "llm-sweep"]
    else:
        experiments_to_run = [args.experiment]

    for exp in experiments_to_run:
        if exp == "direct":
            results = experiment_direct(
                args.dataset, args.n_runs, args.n_ants, args.n_iterations,
                args.n_evals, args.seed, evaluator,
            )
            save_results(results, args.output_dir, f"direct_{args.dataset}")

        elif exp == "transfer":
            results = experiment_transfer(
                args.dataset, args.target_dataset,
                args.n_runs, args.n_ants, args.n_iterations,
                args.n_evals, args.seed, evaluator,
            )
            save_results(results, args.output_dir, f"transfer_{args.dataset}_{args.target_dataset}")

        elif exp == "pheromone":
            results = experiment_pheromone_analysis(
                args.dataset, args.n_ants, args.n_iterations,
                args.n_evals, args.seed, evaluator,
            )
            save_results(results, args.output_dir, f"pheromone_{args.dataset}")

        elif exp == "llm-sweep":
            results = experiment_llm_sweep(
                args.dataset, args.llm, args.n_runs, args.n_ants,
                args.n_iterations, args.n_evals, args.seed, evaluator,
                l1_token_budget=args.l1_tokens,
            )
            save_results(results, args.output_dir, f"llm_sweep_{args.llm}_{args.dataset}")

    logger.info("\nAll experiments completed.")


if __name__ == "__main__":
    main()
