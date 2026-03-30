"""LLaMEA Live Wrapper - Invokes LLaMEA in real-time during NS-SE execution.

Unlike LLaMEAOperatorWrapper which uses pre-trained code, this wrapper
invokes LLaMEA evolution on-demand with a small token budget.

Flow:
1. NS-SE ACO selects ls_intensify_large role
2. LLaMEALiveWrapper.__call__(solution, ctx) is invoked
3. LLaMEA evolves code for this specific instance (small budget: ~5k tokens)
4. Best evolved code is executed to produce a solution
5. Solution is returned to NS-SE

This enables NS-SE to orchestrate WHEN to use LLaMEA's code generation.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from loguru import logger


class LLaMEALiveWrapper:
    """Invokes LLaMEA in real-time as an NS-SE operator.

    When called, runs LLaMEA evolution with a small token budget to generate
    code specialized for the current instance.

    Attributes:
        name: Operator name for NS-SE
        role: NS-SE role (typically ls_intensify_large)
        model: LLM model to use
        max_tokens: Token budget for LLaMEA (default: 5000)
        llm_backend: "openai" or "ollama"
    """

    def __init__(
        self,
        name: str = "llamea_live",
        role: str = "ls_intensify_large",
        model: str = "gpt-4o-mini",
        max_tokens: int = 5000,
        llm_backend: str = "openai",
        api_key: Optional[str] = None,
        eval_timeout: int = 10,
        training_instances: Optional[list[dict]] = None,
    ):
        """Initialize live wrapper.

        Args:
            name: Operator name
            role: NS-SE role
            model: LLM model
            max_tokens: Token budget for LLaMEA evolution
            llm_backend: "openai" or "ollama"
            api_key: API key (uses env var if not provided)
            eval_timeout: Max seconds per code evaluation
            training_instances: List of instances for multi-instance fitness evaluation.
                               Each dict should have: distance_matrix, dimension, optimal (optional)
        """
        self.name = name
        self._role = role
        self.model = model
        self.max_tokens = max_tokens
        self.llm_backend = llm_backend
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.eval_timeout = eval_timeout
        self.training_instances = training_instances or []

        # Cache for evolved code (avoid re-evolving)
        self._code_cache: dict[str, str] = {}  # "evolved" -> code
        self._evolved = False

    @property
    def role(self) -> str:
        return self._role

    @property
    def operator_id(self) -> str:
        return self.name

    def __call__(self, solution: list[int], ctx: Any) -> list[int]:
        """NS-SE operator interface: invoke LLaMEA and return improved solution.

        Args:
            solution: Current solution
            ctx: Execution context with instance data

        Returns:
            Improved solution from LLaMEA, or original if LLaMEA fails
        """
        try:
            # Extract distance matrix
            dm = self._extract_distance_matrix(ctx)
            if dm is None:
                return solution

            n = len(dm)

            # Check cache - only evolve once per problem size
            # Use "evolved" as key to ensure we only evolve once during training
            if self._evolved and self._code_cache:
                # Use the first cached code (should work for all instances of same size)
                code = next(iter(self._code_cache.values()))
                result = self._execute_code(code, dm)
                if result is not None and ctx.valid(result):
                    logger.info(f"[LLaMEA-LIVE] Using cached code for n={n}")
                    return result
                # If execution fails, return original solution (don't re-evolve)
                logger.warning(f"[LLaMEA-LIVE] Cached code failed for n={n}, returning original solution")
                return solution

            # Run LLaMEA evolution
            # Verify instance data
            sample_dist = dm[0][1] if n > 1 else 0
            logger.info(f"[LLaMEA-LIVE] Evolving code for n={n}, d[0][1]={sample_dist:.1f} (budget={self.max_tokens} tokens)...")
            code = self._evolve_code(dm, n)

            if code:
                # Cache the code and mark as evolved (only evolve once)
                self._code_cache[n] = code
                self._evolved = True

                # Execute and return
                result = self._execute_code(code, dm)
                if result is not None and ctx.valid(result):
                    logger.info(f"[LLaMEA-LIVE] Success: evolved code returned valid solution")
                    return result

            return solution

        except Exception as e:
            logger.warning(f"[LLaMEA-LIVE] Error: {e}")
            return solution

    def _extract_distance_matrix(self, ctx: Any) -> Optional[list[list[float]]]:
        """Extract distance matrix from context.

        Supports multiple context formats:
        - TSPContext with _dm attribute
        - Context with instance dict containing distance_matrix
        - Context with instance object having distance_matrix attribute
        """
        # TSPContext stores it as _dm
        if hasattr(ctx, "_dm"):
            return ctx._dm

        # Some contexts have instance attribute
        if hasattr(ctx, "instance"):
            instance = ctx.instance
            if isinstance(instance, dict):
                return instance.get("distance_matrix")
            return getattr(instance, "distance_matrix", None)

        return None

    def _evolve_code(self, dm: list[list[float]], n: int) -> Optional[str]:
        """Run LLaMEA to evolve TSP solver code.

        Args:
            dm: Distance matrix (current instance)
            n: Number of cities

        Returns:
            Best evolved code, or None if failed
        """
        from src.baselines.llamea_wrapper import (
            LLaMEABaseline,
            create_tsp_fitness_wrapper,
            create_tsp_task_prompt,
        )

        # Use multi-instance fitness if training_instances provided
        if self.training_instances:
            fitness_fn = self._create_multi_instance_fitness()
            task_prompt = self._create_multi_instance_task_prompt()
            logger.info(f"[LLaMEA-LIVE] Using multi-instance fitness ({len(self.training_instances)} instances)")
        else:
            # Fallback to single instance
            fitness_fn = create_tsp_fitness_wrapper(dm, timeout_seconds=self.eval_timeout)
            task_prompt = create_tsp_task_prompt(f"instance_n{n}", n)

        # Run LLaMEA with budget
        llamea = LLaMEABaseline(
            model=self.model,
            max_tokens=self.max_tokens,
            llm_backend=self.llm_backend,
            openai_api_key=self.api_key,
            eval_timeout=self.eval_timeout,
            n_parents=5,  # Same as LLaMEA training
            n_offspring=5,
        )

        result = llamea.run(fitness_fn, task_prompt)

        if result.best_code and result.best_fitness < float("inf"):
            logger.info(
                f"[LLaMEA-LIVE] Evolved code in {result.total_tokens} tokens, "
                f"fitness={result.best_fitness:.2f}% (avg gap)"
            )
            return result.best_code

        return None

    def _create_multi_instance_fitness(self) -> Callable[[str], float]:
        """Create fitness function that evaluates on all training instances.

        Returns average gap across all instances (lower is better).
        """
        import signal

        instances = self.training_instances
        timeout = self.eval_timeout

        def timeout_handler(signum, frame):
            raise TimeoutError("Timeout")

        def evaluate_code(code: str) -> float:
            """Evaluate TSP solver code on all instances, return average gap."""
            import random
            import math
            import itertools
            import functools
            import collections

            # Try to compile and extract function
            try:
                exec_globals = {
                    "__builtins__": __builtins__,
                    "random": random,
                    "math": math,
                    "itertools": itertools,
                    "functools": functools,
                    "collections": collections,
                }
                local_vars = {}
                exec(code, exec_globals, local_vars)

                if "solve_tsp" not in local_vars:
                    return float("inf")

                solve_fn = local_vars["solve_tsp"]
            except Exception:
                return float("inf")

            # Evaluate on each instance
            gaps = []
            for inst in instances:
                try:
                    # Run with timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)

                    try:
                        tour = solve_fn(inst["distance_matrix"])
                    finally:
                        signal.alarm(0)

                    # Validate tour
                    n = inst["dimension"]
                    if not isinstance(tour, list) or len(tour) != n or set(tour) != set(range(n)):
                        gaps.append(float("inf"))
                        continue

                    # Calculate tour cost
                    dist = inst["distance_matrix"]
                    cost = sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

                    # Calculate gap
                    optimal = inst.get("optimal")
                    if optimal and optimal > 0:
                        gap = 100.0 * (cost - optimal) / optimal
                    else:
                        gap = cost  # No optimal known

                    gaps.append(gap)

                except Exception:
                    gaps.append(float("inf"))

            # Return average gap (or inf if all failed)
            valid_gaps = [g for g in gaps if g != float("inf")]
            if not valid_gaps:
                return float("inf")

            return sum(valid_gaps) / len(valid_gaps)

        return evaluate_code

    def _create_multi_instance_task_prompt(self) -> str:
        """Create task prompt describing multi-instance TSP task."""
        instance_desc = "\n".join([
            f"  - {inst.get('name', f'inst_{i}')}: {inst['dimension']} cities"
            + (f", optimal={inst['optimal']}" if inst.get('optimal') else "")
            for i, inst in enumerate(self.training_instances)
        ])

        return f"""You are designing a heuristic algorithm for the Traveling Salesman Problem (TSP).

The algorithm will be evaluated on multiple instances:
{instance_desc}

Your task is to write a Python function that takes a distance matrix and returns a tour.
The function should be named `solve_tsp` and have this signature:

```python
def solve_tsp(distance_matrix: list[list[float]]) -> list[int]:
    '''
    Solve TSP instance.

    Args:
        distance_matrix: NxN matrix of distances between cities

    Returns:
        Tour as list of city indices (0 to N-1), visiting each city exactly once
    '''
    # Your implementation here
    pass
```

Requirements:
- Return a valid tour visiting all cities exactly once
- Minimize total tour length (sum of distances along the tour)
- The algorithm should work well across different instance sizes
- You can use standard Python libraries (random, math, itertools, collections)
- Be efficient - the algorithm will be evaluated multiple times

Design a clever heuristic that balances solution quality with computation time.
Write only the function implementation."""

    def _execute_code(self, code: str, dm: list[list[float]]) -> Optional[list[int]]:
        """Execute evolved code to get a solution.

        Args:
            code: Python code with solve_tsp function
            dm: Distance matrix

        Returns:
            Tour, or None if execution failed
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timeout")

        try:
            # Compile code
            namespace: dict = {
                "__builtins__": __builtins__,
                "random": __import__("random"),
                "math": __import__("math"),
            }
            exec(code, namespace)

            if "solve_tsp" not in namespace:
                return None

            solve_fn = namespace["solve_tsp"]

            # Execute with timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.eval_timeout)
            try:
                result = solve_fn(dm)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # Validate
            n = len(dm)
            if isinstance(result, list) and len(result) == n and set(result) == set(range(n)):
                return result

            return None

        except Exception:
            return None

    def __repr__(self) -> str:
        return f"LLaMEALiveWrapper(name={self.name!r}, model={self.model!r}, budget={self.max_tokens})"

    def get_cached_code(self) -> dict[int, str]:
        """Return cached code for persistence.

        Returns:
            Dict mapping problem size (n) to evolved code string.
        """
        return self._code_cache.copy()

    def get_code_for_pool(self) -> Optional[str]:
        """Return the best cached code for saving to operator pool.

        Returns the code for the largest problem size, as it's likely
        the most general.
        """
        if not self._code_cache:
            return None
        # Return code for largest n
        max_n = max(self._code_cache.keys())
        return self._code_cache[max_n]
