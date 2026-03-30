#!/usr/bin/env python3
"""Run NAS Symbolic Executor experiments for the GEAKG paper.

Demonstrates that the KG captures autonomous procedural knowledge:
given only a pheromone snapshot (from ACO training on source dataset)
+ base operators, it generates good GNN architectures with:
  - 0 LLM tokens
  - 0 ACO training on target
  - Pure symbolic graph traversal

Comparison:
  1. NAS Symbolic Executor (pheromone-guided, 0 tokens, 0 ACO)
  2. Regularized Evolution (Real et al. 2019, standard NAS baseline)
  3. ACO cold-start (no transfer, full ACO budget)
  4. Random search (uniform sampling)

Usage:
    python scripts/run_nas_symbolic.py --source cora --targets citeseer,pubmed
    python scripts/run_nas_symbolic.py --source cora --targets citeseer --quick
    python scripts/run_nas_symbolic.py --all-sources
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.domains.nas.graph_evaluator import NASBenchGraphEvaluator
from src.geakg.transfer.nas_symbolic_executor import (
    NASSymbolicExecutor,
    NASExecutionResult,
    build_operators_by_role,
    load_snapshot_from_json,
)


def find_best_snapshot(
    results_dir: str,
    source_dataset: str,
    preferred_llm: str = "gpt-5.2",
) -> str | None:
    """Find the best snapshot JSON for a source dataset.

    Prefers LLM-generated snapshots (gpt-5.2 > gpt-4o-mini > none).
    Among same LLM, picks the most recent file.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    candidates: list[tuple[int, Path]] = []

    llm_priority = {"gpt-5.2": 3, "gpt-4o-mini": 2, "none": 1}

    for f in results_path.glob(f"nasbench_graph_llm_sweep_*_{source_dataset}_*.json"):
        name = f.stem
        # Extract LLM from filename
        for llm_name, priority in llm_priority.items():
            if f"llm_sweep_{llm_name}_{source_dataset}" in name:
                candidates.append((priority, f))
                break

    if not candidates:
        return None

    # Sort by priority (desc), then by modification time (desc)
    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return str(candidates[0][1])


def run_symbolic_executor(
    pheromones: dict[str, float],
    evaluator: NASBenchGraphEvaluator,
    n_evals: int,
    temperature: float = 0.5,
    stagnation_threshold: int = 8,
    seed: int = 42,
    operator_pool: Any = None,
) -> dict[str, Any]:
    """Run the NAS Symbolic Executor with iterative refinement.

    Uses eval-per-step refinement, stagnation detection, and intelligent
    restarts (from best_arch with perturbation).

    Args:
        operator_pool: Optional OperatorPool with L1 operators (gpt-5.2).
            If provided, mixes A₀ + L1 operators (like the full ACO system).
    """
    operators = build_operators_by_role(operator_pool=operator_pool)

    # Scale restarts: ~40 evals per restart gives enough steps to refine
    evals_per_restart = 40
    n_restarts = max(1, (n_evals + evals_per_restart - 1) // evals_per_restart)

    executor = NASSymbolicExecutor(
        pheromones=pheromones,
        operators_by_role=operators,
        n_restarts=n_restarts,
        stagnation_threshold=stagnation_threshold,
        temperature=temperature,
        seed=seed,
    )

    result = executor.execute(evaluator, n_evals_budget=n_evals)

    return {
        "method": "symbolic_executor",
        "best_accuracy": result.best_accuracy,
        "best_architecture": (
            result.best_architecture.to_dict()
            if result.best_architecture
            else None
        ),
        "convergence": result.convergence,
        "n_evals": result.total_evals,
        "n_restarts": result.total_walks,
        "wall_time_s": result.elapsed_time,
        "temperature": temperature,
        "stagnation_threshold": stagnation_threshold,
        "llm_tokens": 0,
        "aco_iterations": 0,
    }


def run_aco_cold_start(
    evaluator: NASBenchGraphEvaluator,
    n_evals: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run ACO cold-start (no transfer) as baseline."""
    from scripts.run_nas_graph_benchmark import geakg_nas_graph

    result = geakg_nas_graph(
        evaluator,
        n_ants=10,
        n_iterations=50,
        seed=seed,
        n_evals_budget=n_evals,
    )
    result["method"] = "aco_cold_start"
    return result


def run_reg_evo(
    evaluator: NASBenchGraphEvaluator,
    n_evals: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Regularized Evolution (Real et al. 2019) as baseline."""
    from scripts.run_nas_graph_benchmark import regularized_evolution_graph

    return regularized_evolution_graph(evaluator, n_evals, seed=seed)


def run_random_search(
    evaluator: NASBenchGraphEvaluator,
    n_evals: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run random search as baseline."""
    from scripts.run_nas_graph_benchmark import random_search_graph

    return random_search_graph(evaluator, n_evals, seed)


def experiment_symbolic_transfer(
    source_dataset: str,
    target_datasets: list[str],
    n_runs: int,
    n_evals: int,
    seed: int,
    results_dir: str,
    temperature: float = 0.5,
) -> dict[str, Any]:
    """Run symbolic executor experiment: source → multiple targets.

    For each target dataset, compares:
    1. Symbolic executor (pheromones from source + L1 operators, 0 tokens, 0 ACO)
    2. Regularized Evolution (Real et al. 2019)
    3. ACO cold-start (full budget, no transfer)
    4. Random search
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"NAS SYMBOLIC EXECUTOR EXPERIMENT")
    logger.info(f"Source: {source_dataset} → Targets: {target_datasets}")
    logger.info(f"Budget: {n_evals} evals, {n_runs} runs, temp={temperature}")
    logger.info(f"{'='*70}")

    # Load pheromone snapshot from source
    snapshot_path = find_best_snapshot(results_dir, source_dataset)
    if snapshot_path is None:
        logger.error(
            f"No snapshot found for {source_dataset} in {results_dir}. "
            f"Run llm-sweep first."
        )
        return {"error": f"no_snapshot_for_{source_dataset}"}

    pheromones, snapshot_meta = load_snapshot_from_json(snapshot_path)
    logger.info(
        f"Loaded snapshot: {snapshot_meta['source_file']}"
    )
    logger.info(
        f"  LLM: {snapshot_meta['llm']}, "
        f"source_acc: {snapshot_meta['best_accuracy']:.2f}%, "
        f"tokens: {snapshot_meta['total_tokens']}, "
        f"edges: {len(pheromones)}"
    )

    # Pick best pheromones across runs
    pheromones = _select_best_pheromones(snapshot_path)

    # Load L1 operator pool (gpt-5.2) if available
    operator_pool = _load_best_operator_pool(results_dir)
    if operator_pool is not None:
        logger.info(
            f"  L1 pool: {operator_pool.total_operators} operators, "
            f"{len(operator_pool.roles)} roles"
        )
    else:
        logger.warning("  No L1 operator pool found — using A₀ only")

    all_experiment_results: dict[str, Any] = {
        "experiment": "nas_symbolic_transfer",
        "source_dataset": source_dataset,
        "snapshot": snapshot_meta,
        "n_pheromone_edges": len(pheromones),
        "temperature": temperature,
        "n_runs": n_runs,
        "n_evals": n_evals,
        "timestamp": datetime.now().isoformat(),
        "targets": {},
    }

    for target in target_datasets:
        logger.info(f"\n--- Target: {target} ---")

        evaluator = NASBenchGraphEvaluator(
            dataset=target, seed=seed,
        )

        target_results: dict[str, list] = defaultdict(list)

        for run in range(n_runs):
            run_seed = seed + run
            logger.info(f"  Run {run + 1}/{n_runs} (seed={run_seed})")

            # 1. Symbolic executor (A₀ + L1 operators + pheromones)
            sym_res = run_symbolic_executor(
                pheromones, evaluator, n_evals,
                temperature=temperature, seed=run_seed,
                operator_pool=operator_pool,
            )
            target_results["symbolic_executor"].append(sym_res)
            logger.info(
                f"    Symbolic: {sym_res['best_accuracy']:.2f}% "
                f"({sym_res['n_evals']} evals, {sym_res['wall_time_s']:.2f}s)"
            )

            # 2. Regularized Evolution
            regevo_res = run_reg_evo(evaluator, n_evals, run_seed)
            target_results["reg_evolution"].append(regevo_res)
            logger.info(
                f"    RegEvo:   {regevo_res['best_accuracy']:.2f}%"
            )

            # 3. ACO cold-start
            aco_res = run_aco_cold_start(evaluator, n_evals, run_seed)
            target_results["aco_cold_start"].append(aco_res)
            logger.info(
                f"    ACO cold: {aco_res['best_accuracy']:.2f}% "
                f"({aco_res.get('n_evals', n_evals)} evals)"
            )

            # 4. Random search
            rand_res = run_random_search(evaluator, n_evals, run_seed)
            target_results["random_search"].append(rand_res)
            logger.info(
                f"    Random:   {rand_res['best_accuracy']:.2f}%"
            )

        # Aggregate
        target_summary = _aggregate_target(target_results, target)
        all_experiment_results["targets"][target] = target_summary

    # Print summary table
    _print_summary_table(all_experiment_results)

    return all_experiment_results


def _select_best_pheromones(json_path: str) -> dict[str, float]:
    """Select pheromones from the run with highest accuracy."""
    data = json.loads(Path(json_path).read_text())
    runs = data.get("runs", [])
    if not runs:
        return {}

    best_run = max(runs, key=lambda r: r.get("best_accuracy", 0))
    return best_run.get("pheromones_display", {})


def _load_best_operator_pool(results_dir: str) -> Any:
    """Load the best L1 operator pool (prefers gpt-5.2).

    Returns:
        OperatorPool or None if no pool found.
    """
    from src.geakg.layers.l1.pool import OperatorPool

    pool_dir = Path(results_dir) / "pools"
    if not pool_dir.exists():
        return None

    # Prefer gpt-5.2 > gpt-4o-mini
    for pattern in ["graph_pool_gpt-5.2_*.json", "graph_pool_gpt-4o-mini_*.json"]:
        candidates = list(pool_dir.glob(pattern))
        if candidates:
            # Pick most recent
            best = max(candidates, key=lambda f: f.stat().st_mtime)
            logger.info(f"  Loading L1 pool: {best}")
            return OperatorPool.load(best)

    return None


def _aggregate_target(
    results: dict[str, list],
    dataset: str,
) -> dict[str, Any]:
    """Aggregate results for a target dataset."""
    summary: dict[str, Any] = {"dataset": dataset, "methods": {}}

    for method, runs in results.items():
        accs = [r["best_accuracy"] for r in runs]
        summary["methods"][method] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "best_accuracy": float(np.max(accs)),
            "worst_accuracy": float(np.min(accs)),
            "n_runs": len(runs),
            "runs": runs,
        }

    # Statistical comparisons
    sym_accs = [r["best_accuracy"] for r in results.get("symbolic_executor", [])]
    rand_accs = [r["best_accuracy"] for r in results.get("random_search", [])]
    regevo_accs = [r["best_accuracy"] for r in results.get("reg_evolution", [])]

    try:
        from scipy import stats as scipy_stats

        if len(sym_accs) >= 2 and len(rand_accs) >= 2:
            _, p_val = scipy_stats.wilcoxon(sym_accs, rand_accs)
            summary["symbolic_vs_random_p"] = float(p_val)

        if len(sym_accs) >= 2 and len(regevo_accs) >= 2:
            _, p_val = scipy_stats.wilcoxon(sym_accs, regevo_accs)
            summary["symbolic_vs_regevo_p"] = float(p_val)
    except Exception:
        pass

    return summary


def _print_summary_table(results: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    source = results.get("source_dataset", "?")
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: NAS Symbolic Executor (source={source})")
    logger.info(f"{'='*80}")
    logger.info(
        f"{'Target':<12} {'Method':<20} {'Mean ± Std':>15} "
        f"{'Best':>8} {'Tokens':>8} {'ACO':>5}"
    )
    logger.info("-" * 72)

    for target, target_data in results.get("targets", {}).items():
        methods = target_data.get("methods", {})
        for method, stats in methods.items():
            mean = stats["mean_accuracy"]
            std = stats["std_accuracy"]
            best = stats["best_accuracy"]

            # Determine tokens and ACO status
            tokens = "0"
            aco = "No"
            if method == "aco_cold_start":
                tokens = "0"
                aco = "Yes"
            elif method == "random_search":
                tokens = "0"
                aco = "No"

            logger.info(
                f"{target:<12} {method:<20} {mean:>6.2f} ± {std:<6.2f} "
                f"{best:>7.2f} {tokens:>8} {aco:>5}"
            )

        # Print p-values if available
        for label, key in [
            ("Symbolic vs Random", "symbolic_vs_random_p"),
            ("Symbolic vs RegEvo", "symbolic_vs_regevo_p"),
        ]:
            p_val = target_data.get(key)
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                logger.info(f"  {'':>12} {label}: p={p_val:.4f} ({sig})")
        logger.info("")


def save_results(results: dict[str, Any], output_dir: str) -> Path:
    """Save results to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = results.get("source_dataset", "unknown")
    filename = f"nas_symbolic_{source}_{timestamp}.json"
    filepath = out_path / filename

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                (str(k) if not isinstance(k, str) else k): _clean(v)
                for k, v in obj.items()
                if k != "pheromones_raw"
            }
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(filepath, "w") as f:
        json.dump(_clean(results), f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NAS Symbolic Executor experiments"
    )
    parser.add_argument(
        "--source", type=str, default="cora",
        help="Source dataset (pheromones trained here)",
    )
    parser.add_argument(
        "--targets", type=str, default="citeseer,pubmed",
        help="Comma-separated target datasets",
    )
    parser.add_argument(
        "--all-sources", action="store_true",
        help="Run all available source datasets",
    )
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--n-evals", type=int, default=200,
        help="Evaluation budget per method",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Softmax temperature (0=greedy, large=uniform)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/nas_bench_graph",
        help="Directory with llm_sweep JSON snapshots",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/nas_bench_graph",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test (fewer runs, smaller budget)",
    )

    args = parser.parse_args()

    if args.quick:
        args.n_runs = 3
        args.n_evals = 50

    targets = [t.strip() for t in args.targets.split(",")]

    if args.all_sources:
        # Run for all available source snapshots
        results_path = Path(args.results_dir)
        sources = set()
        for f in results_path.glob("nasbench_graph_llm_sweep_*.json"):
            parts = f.stem.split("_")
            # Last part before timestamp is dataset
            for ds in ["cora", "citeseer", "pubmed", "cs", "physics",
                        "photo", "computers", "arxiv", "proteins"]:
                if ds in parts:
                    sources.add(ds)

        logger.info(f"Found sources: {sorted(sources)}")

        for source in sorted(sources):
            # Target = all datasets except source
            all_ds = ["cora", "citeseer", "pubmed", "cs", "physics",
                       "photo", "computers", "arxiv", "proteins"]
            tgts = [d for d in all_ds if d != source]

            results = experiment_symbolic_transfer(
                source, tgts, args.n_runs, args.n_evals, args.seed,
                args.results_dir, args.temperature,
            )
            if "error" not in results:
                save_results(results, args.output_dir)
    else:
        results = experiment_symbolic_transfer(
            args.source, targets, args.n_runs, args.n_evals, args.seed,
            args.results_dir, args.temperature,
        )
        if "error" not in results:
            save_results(results, args.output_dir)
        else:
            logger.error(f"Experiment failed: {results['error']}")


if __name__ == "__main__":
    main()
