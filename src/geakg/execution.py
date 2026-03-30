"""Operator Execution - Core logic for applying operators in NS-SE.

This module provides the execution layer that bridges:
- Generic operators (from permutation.py)
- synthesized synthesized operators (from DynamicSynthesisHook)
- Domain-specific evaluation via DomainConfig

Used by the ACO loop to apply operator paths to solutions.

IMPORTANT: This module is domain-agnostic. All domain-specific logic
(validation, fitness, etc.) comes from DomainConfig in src/domains/.
"""

import random
import re
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from src.geakg.generic_operators.permutation import (
    greedy_by_fitness,
    partial_restart,
    random_permutation_construct,
    segment_reverse,
    segment_shuffle,
    swap,
    # Internal helpers used by some operators
    _insert,
    _invert,
)

if TYPE_CHECKING:
    from src.domains.base import DomainConfig


def is_synth_operator(op_name: str) -> bool:
    """Check if an operator is synthesized-synthesized based on naming pattern.

    synthesized operators have a 6-character hex hash suffix like '_3fb322'.
    Generic operators don't have this pattern.

    Args:
        op_name: Operator name to check.

    Returns:
        True if operator appears to be synthesized-synthesized.
    """
    return bool(re.search(r"_[a-f0-9]{6}$", op_name))


def compile_operator_code(code: str) -> Callable | None:
    """Compile operator code string into a callable function.

    Args:
        code: Python source code containing a function definition.

    Returns:
        Compiled function or None if compilation fails.
    """
    try:
        builtin_names = {"math", "random", "itertools", "collections", "copy", "heapq"}
        namespace = {
            "math": __import__("math"),
            "random": __import__("random"),
            "itertools": __import__("itertools"),
            "collections": __import__("collections"),
            "copy": __import__("copy"),
            "heapq": __import__("heapq"),
        }
        exec(code, namespace)

        # Find the user-defined function (not builtins)
        for name, obj in namespace.items():
            if name not in builtin_names and not name.startswith("_"):
                if callable(obj) and not isinstance(obj, type):
                    return obj
        return None
    except Exception as e:
        logger.debug(f"[EXECUTION] Failed to compile operator code: {e}")
        return None


# Track failed operators to disable them after repeated failures
_failed_operator_counts: dict[str, int] = {}
_disabled_operators: set[str] = set()
MAX_FAILURES_BEFORE_DISABLE = 3


def apply_synth_operator(
    op: str,
    solution: Any,
    instance_data: dict[str, Any],
    synthesis_hook: "DynamicSynthesisHook",
    domain_config: "DomainConfig",
    timeout: float | None = None,
    ctx: Any = None,
) -> Any | None:
    """Try to apply an synthesized synthesized operator.

    synthesized operators use agnostic signature: operator(solution, ctx) -> solution
    The ctx (DomainContext) provides cost, delta, neighbors, evaluate, valid methods.

    Args:
        op: Operator ID.
        solution: Current solution (domain-specific type).
        instance_data: Instance data dict (contains solution + problem data).
        synthesis_hook: Hook containing the operator registry.
        domain_config: Domain configuration for validation.
        timeout: Max seconds to wait for operator execution. If None, uses
                 adaptive timeout based on instance size.
        ctx: DomainContext for agnostic operators (created if not provided).

    Returns:
        Modified solution if successful, None if operator not found or failed.
    """
    import signal
    import threading

    # Skip disabled operators immediately
    if op in _disabled_operators:
        return None

    # Get instance size for adaptive timeout
    n = instance_data.get("dimension", len(solution) if hasattr(solution, "__len__") else 100)

    # Adaptive timeout based on instance size
    # Base 5s + 0.1s per element, max 60s
    # This gives ~10s for n=50, ~15s for n=100, max 60s for large instances
    if timeout is None:
        timeout = min(60.0, 5.0 + 0.1 * n)

    def execute_with_timeout(fn: Callable, args: tuple) -> Any:
        """Execute operator function with timeout protection."""
        # Only use signal-based timeout in main thread
        if threading.current_thread() is threading.main_thread():
            def handler(signum, frame):
                raise TimeoutError(f"synthesized operator {op} exceeded {timeout}s")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return fn(*args)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # In non-main thread, just run without timeout
            # (signal doesn't work in threads)
            return fn(*args)

    # Create context if not provided - synthesized operators use (solution, ctx) signature
    if ctx is None:
        ctx = domain_config.create_context(instance_data)

    # Agnostic operator args: (solution, ctx)
    args = (solution, ctx)

    # First, try to get pre-compiled function
    compiled_fn = synthesis_hook.get_compiled_operator(op)
    if compiled_fn is not None:
        try:
            import time as _time
            _t0 = _time.time()
            result = execute_with_timeout(compiled_fn, args)
            _elapsed = _time.time() - _t0
            # Skip slow warning for LLaMEA operators (they invoke LLM)
            if _elapsed > 1.0 and not op.startswith("llamea"):
                logger.warning(f"[EXECUTION] SLOW operator {op}: {_elapsed:.1f}s (n={n})")
            # Validate result using domain config
            if domain_config.validate_solution(result, instance_data):
                synthesis_hook.on_operator_selected(op)
                return result
            else:
                # Invalid result
                _record_failure(op)
                return None
        except TimeoutError as e:
            # Timeouts are also failures - disable after repeated timeouts
            logger.debug(f"[EXECUTION] synthesized operator {op} timeout: {e}")
            _record_failure(op)
            return None
        except Exception as e:
            # Real errors (invalid results, crashes) should be recorded
            logger.debug(f"[EXECUTION] synthesized operator {op} failed: {e}")
            _record_failure(op)
            return None

    # Try to get the operator record and recompile if needed
    record = synthesis_hook.registry.get(op)
    if record is not None and record.code:
        compiled_fn = compile_operator_code(record.code)
        if compiled_fn:
            try:
                result = execute_with_timeout(compiled_fn, args)
                if domain_config.validate_solution(result, instance_data):
                    synthesis_hook.on_operator_selected(op)
                    return result
                else:
                    # Invalid result
                    _record_failure(op)
                    return None
            except TimeoutError as e:
                # Timeouts are also failures - disable after repeated timeouts
                logger.debug(f"[EXECUTION] synthesized operator {op} timeout: {e}")
                _record_failure(op)
                return None
            except Exception as e:
                logger.debug(f"[EXECUTION] synthesized operator {op} recompile failed: {e}")
                _record_failure(op)
                return None

    return None


def _record_failure(op: str) -> None:
    """Record a failure for an operator and disable if threshold reached."""
    _failed_operator_counts[op] = _failed_operator_counts.get(op, 0) + 1
    if _failed_operator_counts[op] >= MAX_FAILURES_BEFORE_DISABLE:
        _disabled_operators.add(op)
        logger.warning(
            f"[EXECUTION] Disabled synthesized operator {op} after {MAX_FAILURES_BEFORE_DISABLE} failures"
        )


def reset_disabled_operators() -> None:
    """Reset disabled operators tracking. Call at start of new experiment."""
    _failed_operator_counts.clear()
    _disabled_operators.clear()


def get_disabled_operators() -> set[str]:
    """Get the set of disabled operators."""
    return _disabled_operators.copy()


def apply_operator(
    op: str,
    solution: Any,
    instance_data: dict[str, Any],
    domain_config: "DomainConfig",
    synthesis_hook: "DynamicSynthesisHook | None" = None,
    ctx: Any = None,
) -> Any:
    """Apply an operator to a solution (domain-agnostic).

    This is the main entry point for operator execution. It handles:
    1. synthesized synthesized operators (if synthesis_hook provided)
    2. Generic operators based on representation type

    All operators now use DomainContext (ctx) for evaluation, following
    Lampson's principle "Keep secrets" - domain details are hidden.

    Args:
        op: Operator name/ID.
        solution: Current solution (domain-specific type).
        instance_data: Instance data dict.
        domain_config: Domain configuration.
        synthesis_hook: Optional hook for synthesized synthesized operators.
        ctx: Optional pre-created DomainContext (created if not provided).

    Returns:
        Modified solution.
    """
    # Create context if not provided (cache for efficiency)
    if ctx is None:
        ctx = domain_config.create_context(instance_data)

    # First, try synthesized synthesized operator
    if synthesis_hook is not None:
        result = apply_synth_operator(op, solution, instance_data, synthesis_hook, domain_config, ctx=ctx)
        if result is not None:
            return result

    # Fall back to generic operators based on representation
    # Dispatch by representation type
    representation = getattr(domain_config, 'representation_type', None)
    if representation is not None:
        rep_value = representation.value if hasattr(representation, 'value') else str(representation)
        if rep_value == "architecture_dag":
            return _apply_architecture_operator(op, solution, ctx)

    return _apply_permutation_operator(op, solution, ctx, synthesis_hook)


def _apply_permutation_operator(
    op: str,
    solution: list[int],
    ctx: Any,
    synthesis_hook: "DynamicSynthesisHook | None" = None,
) -> list[int]:
    """Apply a generic permutation operator using DomainContext.

    Args:
        op: Operator name.
        solution: Current permutation.
        ctx: DomainContext with 5 methods (cost, delta, neighbors, evaluate, valid).
        synthesis_hook: Optional hook (for logging).

    Returns:
        Modified permutation.
    """
    n = len(solution)
    solution = solution.copy()

    # === CONSTRUCTION OPERATORS ===
    if op == "greedy_by_fitness":
        # Need a partial fitness function - use ctx.evaluate on partial tour
        # NOTE: Don't use ctx.valid() here - partial tours are not valid permutations
        # but we still need to evaluate them for greedy construction
        def partial_fitness(perm: list[int], candidate: int) -> float:
            if not perm:
                return 0.0
            # Evaluate partial tour cost (last edge from current end to candidate)
            # This is the greedy heuristic: minimize distance to next city
            last_city = perm[-1]
            return ctx._dm[last_city][candidate]  # Direct distance lookup

        return greedy_by_fitness(n, partial_fitness)

    elif op == "random_insertion":
        result = [solution[0]] if solution else []
        for elem in solution[1:]:
            pos = random.randint(0, len(result))
            result.insert(pos, elem)
        return result

    elif op == "pairwise_merge":
        result = solution.copy()
        random.shuffle(result)
        return result

    elif op == "random_permutation":
        return random_permutation_construct(n)

    # === LOCAL SEARCH OPERATORS ===
    elif op == "swap":
        if n > 1:
            return swap(solution)
        return solution

    elif op == "segment_reverse":
        if n > 3:
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)
            return segment_reverse(solution, i, j)
        return solution

    elif op == "variable_depth_search":
        best = solution.copy()
        best_cost = ctx.evaluate(best)

        for depth in range(1, 4):
            current = best.copy()
            for _ in range(depth * 5):
                candidate = swap(current)
                candidate_cost = ctx.evaluate(candidate)
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    current = candidate
        return best

    elif op == "vnd_generic":
        best = solution.copy()
        best_cost = ctx.evaluate(best)

        neighborhoods = [swap, _insert, _invert]
        improved = True
        max_iters = 20
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1
            for neighborhood in neighborhoods:
                candidate = neighborhood(best)
                candidate_cost = ctx.evaluate(candidate)
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
        return best

    # === PERTURBATION OPERATORS ===
    elif op == "segment_shuffle":
        return segment_shuffle(solution, k=max(3, n // 5))

    elif op == "partial_restart":
        return partial_restart(solution, ratio=0.3)

    elif op == "history_guided_perturb":
        # Without history, just do a random perturbation
        return segment_shuffle(solution, k=max(3, n // 5))

    # === UNKNOWN OPERATOR ===
    else:
        if synthesis_hook is not None and is_synth_operator(op):
            # synthesized operator was already tried by apply_synth_operator above
            # If it's disabled, don't log anything (already logged when disabled)
            # If it's in failed counts, use debug level to reduce noise
            if op in _disabled_operators:
                pass  # Silent - already logged when disabled
            elif op in _failed_operator_counts:
                logger.debug(f"[EXECUTION] synthesized operator '{op}' failed (attempt {_failed_operator_counts[op]})")
            else:
                record = synthesis_hook.registry.get(op)
                if record is not None:
                    logger.debug(
                        f"[EXECUTION] synthesized operator '{op}' execution failed "
                        f"(compiled_fn={'SET' if record.compiled_fn else 'NONE'})"
                    )
                else:
                    logger.debug(f"[EXECUTION] Unknown synthesized operator '{op}' not in registry")
        else:
            logger.debug(f"[EXECUTION] Unknown operator '{op}', returning unchanged")
        return solution


def _apply_architecture_operator(
    op: str,
    solution: Any,
    ctx: Any,
) -> Any:
    """Apply an architecture operator for NAS.

    Architecture operators modify neural architecture specifications.
    They use the NASContext interface for validation and evaluation.

    Args:
        op: Operator name.
        solution: Current architecture specification.
        ctx: NASContext with evaluate, valid, random_solution methods.

    Returns:
        Modified architecture specification.
    """
    try:
        from src.domains.nas.operators import apply_nas_operator
        result = apply_nas_operator(op, solution, ctx)
        if result is not None:
            return result
    except ImportError:
        logger.debug(f"[EXECUTION] NAS operators not available for {op}")
    except Exception as e:
        logger.debug(f"[EXECUTION] NAS operator {op} failed: {e}")

    return solution


def evaluate_operator_path(
    operator_path: list[str],
    instance_data: dict[str, Any],
    domain_config: "DomainConfig",
    synthesis_hook: "DynamicSynthesisHook | None" = None,
) -> float:
    """Evaluate a sequence of operators on a problem instance.

    Starts with a random solution and applies each operator in sequence.
    Uses DomainContext for efficient evaluation (ctx is created once and reused).

    Args:
        operator_path: List of operator names to apply.
        instance_data: Instance data dict (domain-specific, includes problem data).
        domain_config: Domain configuration.
        synthesis_hook: Optional hook for synthesized synthesized operators.

    Returns:
        Final solution cost after applying all operators.
    """
    # Generate random initial solution (permutation of indices)
    dimension = instance_data.get("dimension", 30)
    solution = list(range(dimension))
    random.shuffle(solution)

    # Build instance data with the random solution
    eval_data = instance_data.copy()
    eval_data["tour"] = solution  # Add random tour to instance data

    # Create context once and reuse for efficiency
    ctx = domain_config.create_context(eval_data)

    # Apply each operator (ctx is passed to avoid recreating it)
    for op in operator_path:
        solution = apply_operator(op, solution, eval_data, domain_config, synthesis_hook, ctx=ctx)
        eval_data["tour"] = solution  # Update tour in data

    return ctx.evaluate(solution)


def evaluate_operator_path_with_stats(
    operator_path: list[str],
    instance_data: dict[str, Any],
    domain_config: "DomainConfig",
    synthesis_hook: "DynamicSynthesisHook | None" = None,
) -> tuple[float, dict[str, list[float]]]:
    """Evaluate operator path and return per-operator fitness deltas.

    Same as evaluate_operator_path but tracks the fitness change (delta)
    for each operator application. This enables evolutionary feedback
    showing which operators actually improve solutions.

    Note: Construction operators (const_*) are excluded from delta tracking
    because they create new solutions rather than improving existing ones.

    Args:
        operator_path: List of operator names to apply.
        instance_data: Instance data dict (domain-specific, includes problem data).
        domain_config: Domain configuration.
        synthesis_hook: Optional hook for synthesized operators.

    Returns:
        Tuple of (final_fitness, operator_deltas) where operator_deltas maps
        operator names to lists of fitness deltas (positive = improvement).
    """
    # Generate random initial solution (permutation of indices)
    dimension = instance_data.get("dimension", 30)
    solution = list(range(dimension))
    random.shuffle(solution)

    # Build instance data with the random solution
    eval_data = instance_data.copy()
    eval_data["tour"] = solution

    # Create context once and reuse for efficiency
    ctx = domain_config.create_context(eval_data)

    # Track deltas per operator (excluding construction operators)
    operator_deltas: dict[str, list[float]] = {}

    # Apply each operator and track fitness changes
    for op in operator_path:
        fitness_before = ctx.evaluate(solution)
        solution = apply_operator(op, solution, eval_data, domain_config, synthesis_hook, ctx=ctx)
        eval_data["tour"] = solution
        fitness_after = ctx.evaluate(solution)

        # Skip construction operators - delta doesn't make sense for them
        if op.startswith("const_"):
            continue

        # Delta = before - after (positive means improvement, i.e., cost decreased)
        delta = fitness_before - fitness_after
        if op not in operator_deltas:
            operator_deltas[op] = []
        operator_deltas[op].append(delta)

    return ctx.evaluate(solution), operator_deltas


# Export public API
__all__ = [
    "apply_operator",
    "apply_synth_operator",
    "compile_operator_code",
    "evaluate_operator_path",
    "evaluate_operator_path_with_stats",
    "get_disabled_operators",
    "is_synth_operator",
    "reset_disabled_operators",
]
