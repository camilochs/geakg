"""Mini F-Race for statistical operator selection.

This module provides a simplified F-Race implementation that:
1. Evaluates candidates on instances incrementally
2. Uses Wilcoxon signed-rank test to compare
3. Eliminates statistically worse candidates early
4. Saves evaluations by stopping when survivors are few

No external dependencies (irace) - just scipy.stats.wilcoxon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:
    from src.geakg.layers.l1.pool import Operator


def mini_frace(
    candidates: list["Operator"],
    instances: list[Any],
    evaluate_fn: Callable[["Operator", Any], float],
    significance: float = 0.05,
    min_survivors: int = 2,
) -> list["Operator"]:
    """Mini F-Race: eliminate statistically worse candidates.

    Algorithm:
    1. Evaluate all candidates on instance_1
    2. Compare each candidate vs best with Wilcoxon test
    3. Eliminate candidates significantly worse (p < significance)
    4. Repeat with instance_2, instance_3, ...
    5. Stop when survivors <= min_survivors or instances exhausted

    Args:
        candidates: List of operators to race
        instances: List of problem instances for evaluation
        evaluate_fn: Function(operator, instance) -> fitness (lower is better)
        significance: p-value threshold for elimination (default 0.05)
        min_survivors: Stop racing when this many survivors remain

    Returns:
        List of surviving operators, sorted by average fitness
    """
    if len(candidates) <= min_survivors:
        return candidates

    survivors = list(candidates)
    results: dict[str, list[float]] = {c.id: [] for c in survivors}

    logger.info(f"[F-Race] Starting with {len(candidates)} candidates, {len(instances)} instances")

    for i, instance in enumerate(instances):
        if len(survivors) <= min_survivors:
            logger.info(f"[F-Race] Stopping early: {len(survivors)} survivors remain")
            break

        # Evaluate all survivors on this instance
        for candidate in survivors:
            try:
                fitness = evaluate_fn(candidate, instance)
                results[candidate.id].append(fitness)
            except Exception as e:
                logger.warning(f"[F-Race] Evaluation failed for {candidate.name}: {e}")
                results[candidate.id].append(float("inf"))

        # Eliminate losers after sufficient data
        if i >= 2:  # Need at least 3 samples for Wilcoxon
            survivors = _eliminate_losers(survivors, results, significance)
            logger.debug(f"[F-Race] Instance {i+1}: {len(survivors)} survivors")

    # Sort by average fitness
    survivors.sort(key=lambda c: _avg_fitness(results[c.id]))

    logger.info(
        f"[F-Race] Complete: {len(survivors)} survivors from {len(candidates)} candidates"
    )

    # Update fitness scores on operators
    for candidate in survivors:
        candidate.fitness_scores = results[candidate.id]

    return survivors


def _eliminate_losers(
    candidates: list["Operator"],
    results: dict[str, list[float]],
    significance: float,
) -> list["Operator"]:
    """Eliminate candidates significantly worse than the best.

    Uses Wilcoxon signed-rank test (paired, non-parametric).

    Args:
        candidates: Current survivors
        results: Fitness results per candidate
        significance: p-value threshold

    Returns:
        Reduced list of survivors
    """
    if len(candidates) <= 2:
        return candidates

    # Find the best candidate (lowest average fitness)
    best = min(candidates, key=lambda c: _avg_fitness(results[c.id]))
    best_results = results[best.id]

    survivors = [best]

    for candidate in candidates:
        if candidate.id == best.id:
            continue

        cand_results = results[candidate.id]

        # Skip if not enough paired observations
        if len(cand_results) < 3 or len(best_results) < 3:
            survivors.append(candidate)
            continue

        # Wilcoxon test: is candidate significantly worse than best?
        try:
            p_value = _wilcoxon_test(cand_results, best_results)

            if p_value >= significance:
                # Not significantly worse - keep it
                survivors.append(candidate)
            else:
                logger.debug(
                    f"[F-Race] Eliminating {candidate.name} (p={p_value:.4f})"
                )
        except Exception:
            # If test fails, keep the candidate
            survivors.append(candidate)

    return survivors


def _wilcoxon_test(x: list[float], y: list[float]) -> float:
    """Perform Wilcoxon signed-rank test.

    Tests if x is significantly greater than y (one-sided).

    Args:
        x: First sample (candidate results)
        y: Second sample (best results)

    Returns:
        p-value for alternative='greater'
    """
    try:
        from scipy.stats import wilcoxon

        # Compute differences
        n = min(len(x), len(y))
        diffs = [x[i] - y[i] for i in range(n)]

        # Remove zeros (ties)
        diffs = [d for d in diffs if abs(d) > 1e-10]

        if len(diffs) < 3:
            return 1.0  # Not enough data

        _, p_value = wilcoxon(diffs, alternative="greater")
        return p_value

    except ImportError:
        logger.warning("[F-Race] scipy not available, using simple comparison")
        return _simple_comparison(x, y)


def _simple_comparison(x: list[float], y: list[float]) -> float:
    """Simple comparison when scipy is not available.

    Returns pseudo p-value based on win rate.
    """
    n = min(len(x), len(y))
    wins = sum(1 for i in range(n) if x[i] > y[i])
    win_rate = wins / n if n > 0 else 0.5

    # Convert win rate to pseudo p-value
    # High win rate (x worse than y) -> low p-value
    return 1.0 - win_rate


def _avg_fitness(scores: list[float]) -> float:
    """Calculate average fitness, handling empty lists."""
    if not scores:
        return float("inf")
    return sum(scores) / len(scores)


# =============================================================================
# BATCH EVALUATION HELPERS
# =============================================================================


def evaluate_operator_on_instances(
    operator: "Operator",
    instances: list[Any],
    evaluate_fn: Callable[["Operator", Any], float],
) -> list[float]:
    """Evaluate an operator on multiple instances.

    Args:
        operator: Operator to evaluate
        instances: List of problem instances
        evaluate_fn: Evaluation function

    Returns:
        List of fitness scores
    """
    scores = []
    for instance in instances:
        try:
            score = evaluate_fn(operator, instance)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            scores.append(float("inf"))
    return scores


def rank_operators(
    operators: list["Operator"],
    instances: list[Any],
    evaluate_fn: Callable[["Operator", Any], float],
) -> list["Operator"]:
    """Rank operators by average fitness across instances.

    Simple alternative to F-Race for small pools.

    Args:
        operators: Operators to rank
        instances: Evaluation instances
        evaluate_fn: Evaluation function

    Returns:
        Operators sorted by average fitness (best first)
    """
    for op in operators:
        op.fitness_scores = evaluate_operator_on_instances(op, instances, evaluate_fn)

    return sorted(operators, key=lambda op: op.avg_fitness)
