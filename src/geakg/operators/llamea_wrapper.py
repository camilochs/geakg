"""LLaMEA to NS-SE operator wrapper.

Adapts LLaMEA evolved functions (solve_tsp(distance_matrix) -> tour)
to NS-SE operator interface (solution, ctx) -> solution, enabling:
- Pheromone-based selection in SymbolicExecutor
- Automatic transfer to other domains via existing adapters
- Timeout protection and safe fallback
"""

from __future__ import annotations

import signal
from typing import Any, Callable, Optional


class OperatorTimeoutError(Exception):
    """Raised when operator execution exceeds timeout."""

    pass


def run_with_timeout(func: Callable, args: tuple, timeout: float) -> Any:
    """Execute function with timeout using signals (Unix only).

    Args:
        func: Function to execute
        args: Arguments to pass to function
        timeout: Maximum execution time in seconds

    Returns:
        Function result

    Raises:
        OperatorTimeoutError: If function exceeds timeout
    """

    def handler(signum, frame):
        raise OperatorTimeoutError("Function timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func(*args)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class LLaMEAOperatorWrapper:
    """Adapts LLaMEA function to NS-SE interface (solution, ctx) -> solution.

    Allows integrating LLaMEA evolved code as an operator in the
    SymbolicExecutor, participating in pheromone-based selection and
    being transferable to other domains.

    Attributes:
        llamea_fn: The wrapped LLaMEA function
        name: Operator name for debugging and pheromone tracking
        role: NS-SE role (const_llamea, ls_llamea, etc.)
        timeout: Maximum execution time in seconds

    Example:
        >>> def solve_tsp(dm): return list(range(len(dm)))  # trivial
        >>> wrapper = LLaMEAOperatorWrapper(solve_tsp, "llamea_10k", "const_llamea")
        >>> result = wrapper(initial_tour, ctx)
    """

    def __init__(
        self,
        llamea_fn: Callable[[list[list[float]]], list[int]],
        name: str = "llamea_operator",
        role: str = "const_llamea",
        timeout: float = 1.0,
    ):
        """Initialize wrapper.

        Args:
            llamea_fn: LLaMEA function with signature solve_tsp(distance_matrix) -> tour
            name: Operator name (for debugging and pheromones)
            role: NS-SE role (const_llamea, ls_llamea, pert_llamea, etc.)
            timeout: Maximum execution time in seconds
        """
        self.llamea_fn = llamea_fn
        self.name = name
        self._role = role
        self.timeout = timeout

    @property
    def role(self) -> str:
        """NS-SE role for this operator."""
        return self._role

    @property
    def operator_id(self) -> str:
        """Unique identifier for pheromone tracking."""
        return self.name

    @property
    def adapted_fn(self) -> Callable:
        """Return self as the adapted callable."""
        return self

    def __call__(self, solution: list[int], ctx: Any) -> list[int]:
        """NS-SE operator interface: (solution, ctx) -> solution.

        Args:
            solution: Current solution (may be ignored by construction operators)
            ctx: Execution context with instance data and validation methods

        Returns:
            New solution if LLaMEA succeeds and result is valid,
            original solution otherwise (safe fallback)
        """
        try:
            # 1. Extract distance_matrix from context
            distance_matrix = self._extract_distance_matrix(ctx)
            if distance_matrix is None:
                return solution

            # 2. Execute LLaMEA with timeout
            try:
                result = run_with_timeout(
                    self.llamea_fn,
                    args=(distance_matrix,),
                    timeout=self.timeout,
                )
            except OperatorTimeoutError:
                return solution

            # 3. Validate result
            if result is None:
                return solution

            if ctx.valid(result):
                return result

            return solution

        except Exception:
            return solution

    def _extract_distance_matrix(self, ctx: Any) -> Optional[list[list[float]]]:
        """Extract distance matrix from context.

        Handles different context/instance structures.

        Args:
            ctx: Execution context

        Returns:
            Distance matrix or None if not found
        """
        if not hasattr(ctx, "instance"):
            return None

        instance = ctx.instance

        # Handle dict instance
        if isinstance(instance, dict):
            return instance.get("distance_matrix")

        # Handle object instance
        return getattr(instance, "distance_matrix", None)

    def __repr__(self) -> str:
        return f"LLaMEAOperatorWrapper(name={self.name!r}, role={self._role!r})"


def load_llamea_as_operator(
    code_path: str,
    name: Optional[str] = None,
    role: str = "const_llamea",
    timeout: float = 1.0,
) -> LLaMEAOperatorWrapper:
    """Load LLaMEA code from file and return NS-SE wrapper.

    Args:
        code_path: Path to .py file containing solve_tsp() function
        name: Operator name (default: filename stem)
        role: NS-SE role
        timeout: Timeout in seconds

    Returns:
        LLaMEAOperatorWrapper ready to use in SymbolicExecutor

    Raises:
        ValueError: If solve_tsp function not found in file
        FileNotFoundError: If code_path doesn't exist
    """
    import importlib.util
    from pathlib import Path

    path = Path(code_path)
    if name is None:
        name = path.stem

    # Load module dynamically
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {code_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find solve_tsp function
    if hasattr(module, "solve_tsp"):
        fn = module.solve_tsp
    else:
        raise ValueError(f"No solve_tsp function found in {code_path}")

    return LLaMEAOperatorWrapper(fn, name, role, timeout)


def create_llamea_operator(
    llamea_fn: Callable[[list[list[float]]], list[int]],
    name: str = "llamea_operator",
    role: str = "const_llamea",
    timeout: float = 1.0,
) -> LLaMEAOperatorWrapper:
    """Create NS-SE operator from LLaMEA function directly.

    Convenience function for when you already have the function in memory.

    Args:
        llamea_fn: LLaMEA function with signature solve_tsp(distance_matrix) -> tour
        name: Operator name
        role: NS-SE role
        timeout: Timeout in seconds

    Returns:
        LLaMEAOperatorWrapper ready to use in SymbolicExecutor
    """
    return LLaMEAOperatorWrapper(llamea_fn, name, role, timeout)
