"""Safe execution of LLM-generated operators with timeout protection.

L1 operators are synthesized by LLMs and may contain bugs like infinite loops
(e.g., a while loop whose condition variable is never updated). This module
provides a thread-based timeout wrapper that terminates stuck operators cleanly.

Usage:
    from src.geakg.safe_exec import safe_call_operator

    new_arch = safe_call_operator(operator, arch, ctx, timeout=2.0)
    if new_arch is None:
        # operator failed or timed out
"""

from __future__ import annotations

import threading
from typing import Any, Callable


def safe_call_operator(
    operator: Any,
    solution: Any,
    ctx: Any,
    *,
    timeout: float = 2.0,
) -> Any | None:
    """Execute an operator with a timeout guard.

    Handles both GenericOperator objects (with .function attribute)
    and raw callables. Returns None if the operator times out, raises
    an exception, or returns a non-architecture result.

    Args:
        operator: A GenericOperator or callable ``f(solution, ctx) -> solution``.
        solution: The current architecture / solution.
        ctx: Execution context (may be None for symbolic executors).
        timeout: Maximum seconds to wait. Default 2s — far more than any
                 legitimate mutation should need.

    Returns:
        The new solution, or None on failure/timeout.
    """
    fn: Callable = getattr(operator, "function", operator)

    result_box: list[Any] = []
    error_box: list[Exception] = []

    def _target() -> None:
        try:
            result_box.append(fn(solution, ctx))
        except Exception as exc:  # noqa: BLE001
            error_box.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Operator is stuck — daemon thread will be garbage-collected.
        return None

    if error_box:
        return None

    return result_box[0] if result_box else None
