"""LLaMEA baseline wrapper for fair comparison.

Uses the official LLaMEA library with Ollama or OpenAI API.
Same model and TOKEN BUDGET as NS-SE for fair comparison.
"""

import os
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class LLaMEAResult(BaseModel):
    """Result from LLaMEA run."""

    best_fitness: float
    best_code: str = ""
    evaluations: int = 0
    llm_calls: int = 0
    wall_time_seconds: float = 0.0
    llm_time_seconds: float = 0.0
    compute_time_seconds: float = 0.0
    llm_failures: int = 0
    llm_success_rate: float = 0.0
    fitness_history: list[float] = Field(default_factory=list)
    # Token tracking for fair comparison with NS-SE
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    stopped_by_token_limit: bool = False


class LLaMEABaseline:
    """Wrapper for LLaMEA baseline.

    Uses the official LLaMEA library (MIT License, Humies 2025 Silver).
    Configured to use Ollama or OpenAI API for fair comparison.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        budget: int = 1000,
        max_tokens: int | None = None,
        n_parents: int = 5,
        n_offspring: int = 5,
        ollama_host: str = "http://localhost:11434",
        eval_timeout: int = 60,
        llm_backend: Literal["ollama", "openai"] = "ollama",
        openai_api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize LLaMEA baseline.

        Args:
            model: Model name (same as NS-SE for fairness)
            budget: Total fitness evaluations budget (ignored if max_tokens set)
            max_tokens: Token budget limit (for fair comparison with NS-SE)
            n_parents: Number of parents per generation
            n_offspring: Number of offspring per generation
            ollama_host: Ollama API host
            eval_timeout: Max seconds per evaluation (default 60)
            llm_backend: "ollama" or "openai"
            openai_api_key: OpenAI API key (required if llm_backend="openai")
            reasoning_effort: For GPT-5 models. Options: none, minimal, low, medium, high, xhigh
        """
        self.model = model
        self.budget = budget
        self.max_tokens = max_tokens
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.ollama_host = ollama_host
        self.eval_timeout = eval_timeout
        self.llm_backend = llm_backend
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.reasoning_effort = reasoning_effort

    def run(
        self,
        fitness_function: Callable[[str], float],
        task_prompt: str,
    ) -> LLaMEAResult:
        """Run LLaMEA optimization.

        Args:
            fitness_function: Function that evaluates code string -> fitness
            task_prompt: Description of the task for LLM

        Returns:
            LLaMEAResult with best solution and statistics
        """
        try:
            from llamea import LLaMEA
            from llamea.llm import Ollama_LLM, OpenAI_LLM
        except ImportError:
            raise ImportError(
                "LLaMEA package required. "
                "Install with: uv add llamea"
            )

        start_time = time.time()

        # Create LLM client based on backend
        if self.llm_backend == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for openai backend")
            base_llm = OpenAI_LLM(model=self.model, api_key=self.openai_api_key)
        else:
            base_llm = Ollama_LLM(model=self.model)

        # Use a mutable container to track LLM time and tokens across closure
        stats = {
            "llm_time": 0.0,
            "eval_count": 0,
            "failure_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "token_limit_reached": False,
        }
        fitness_history: list[float] = []

        # Track Ollama module reference for cleanup
        ollama_module = None
        original_ollama_chat = None

        # Wrap the OpenAI client directly to capture token usage
        # LLaMEA's query() method discards usage info, so we need to intercept at client level
        if self.llm_backend == "openai" and hasattr(base_llm, "client"):
            original_create = base_llm.client.chat.completions.create

            def tracked_create(*args, **kwargs):
                # Check token limit before making call
                if self.max_tokens and stats["prompt_tokens"] + stats["completion_tokens"] >= self.max_tokens:
                    stats["token_limit_reached"] = True
                    raise StopIteration("Token budget exhausted")

                # gpt-5 models only support temperature=1.0
                model_name = kwargs.get("model", "")
                if "gpt-5" in model_name:
                    if "temperature" in kwargs:
                        kwargs["temperature"] = 1.0
                    # Add reasoning_effort for GPT-5 models
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort

                t0 = time.time()
                response = original_create(*args, **kwargs)
                stats["llm_time"] += time.time() - t0
                stats["llm_calls"] += 1

                # Extract REAL token usage from OpenAI response
                if hasattr(response, "usage") and response.usage:
                    stats["prompt_tokens"] += response.usage.prompt_tokens or 0
                    stats["completion_tokens"] += response.usage.completion_tokens or 0

                # Log progress
                total = stats["prompt_tokens"] + stats["completion_tokens"]
                pct = 100.0 * total / self.max_tokens if self.max_tokens else 0
                print(f"  [LLM #{stats['llm_calls']}] +{response.usage.prompt_tokens or 0}+{response.usage.completion_tokens or 0} = {total:,} tokens ({pct:.1f}%)")

                return response

            base_llm.client.chat.completions.create = tracked_create
        else:
            # For Ollama: intercept ollama.chat directly to get REAL token counts
            # The response contains prompt_eval_count and eval_count fields
            try:
                import ollama
                ollama_module = ollama
                original_ollama_chat = ollama_module.chat

                def tracked_ollama_chat(*args, **kwargs):
                    # Check token limit before making call
                    if self.max_tokens and stats["prompt_tokens"] + stats["completion_tokens"] >= self.max_tokens:
                        stats["token_limit_reached"] = True
                        raise StopIteration("Token budget exhausted")

                    t0 = time.time()
                    response = original_ollama_chat(*args, **kwargs)
                    stats["llm_time"] += time.time() - t0
                    stats["llm_calls"] += 1

                    # Extract REAL token counts from Ollama response
                    # Ollama returns: prompt_eval_count (input tokens), eval_count (output tokens)
                    prompt_tokens = response.get("prompt_eval_count", 0) or 0
                    completion_tokens = response.get("eval_count", 0) or 0
                    stats["prompt_tokens"] += prompt_tokens
                    stats["completion_tokens"] += completion_tokens

                    # Log progress with REAL token counts
                    total = stats["prompt_tokens"] + stats["completion_tokens"]
                    pct = 100.0 * total / self.max_tokens if self.max_tokens else 0
                    print(f"  [LLM #{stats['llm_calls']}] +{prompt_tokens}+{completion_tokens} = {total:,} tokens ({pct:.1f}%)")

                    return response

                # Monkey-patch ollama.chat
                ollama_module.chat = tracked_ollama_chat
            except ImportError:
                # Fallback if ollama module not available: wrap query method with estimation
                original_query = base_llm.query

                def timed_query_with_estimation(*args, **kwargs):
                    if self.max_tokens and stats["prompt_tokens"] + stats["completion_tokens"] >= self.max_tokens:
                        stats["token_limit_reached"] = True
                        raise StopIteration("Token budget exhausted")

                    t0 = time.time()
                    result = original_query(*args, **kwargs)
                    stats["llm_time"] += time.time() - t0
                    stats["llm_calls"] += 1

                    # Estimate tokens (~4 chars per token)
                    prompt_text = str(args[0]) if args else ""
                    response_text = str(result) if result else ""
                    prompt_est = len(prompt_text) // 4
                    completion_est = len(response_text) // 4
                    stats["prompt_tokens"] += prompt_est
                    stats["completion_tokens"] += completion_est

                    # Log progress (similar to OpenAI)
                    total = stats["prompt_tokens"] + stats["completion_tokens"]
                    pct = 100.0 * total / self.max_tokens if self.max_tokens else 0
                    print(f"  [LLM #{stats['llm_calls']}] ~{prompt_est}+{completion_est} = {total:,} tokens ({pct:.1f}%) [estimated]")

                    return result

                base_llm.query = timed_query_with_estimation

        def tracked_fitness(individual: Any, logger: Any = None) -> Any:
            """Fitness function compatible with LLaMEA's signature.

            LLaMEA passes a Solution object and a logger.
            We extract the code from individual.code and evaluate it.
            """
            stats["eval_count"] += 1
            # Extract code from Solution object (LLaMEA uses .code attribute)
            code = individual.code if hasattr(individual, "code") else str(individual)
            fitness = fitness_function(code)
            fitness_history.append(fitness)
            if fitness == float("inf"):
                stats["failure_count"] += 1
            # Update the individual's fitness and return it
            individual.fitness = fitness
            return individual

        # Create LLaMEA instance
        # Disable logging to avoid directory creation issues with OpenAI backend
        es = LLaMEA(
            f=tracked_fitness,
            llm=base_llm,
            n_parents=self.n_parents,
            n_offspring=self.n_offspring,
            budget=self.budget,
            elitism=True,
            task_prompt=task_prompt,
            minimization=True,  # TSP is a minimization problem
            max_workers=1,  # Disable parallelism so stats tracking works
            eval_timeout=self.eval_timeout,
            log=False,  # Disable logging to avoid dir issues
        )

        # Run evolution (may stop early if token limit reached)
        try:
            best = es.run()
        except StopIteration:
            # Token budget exhausted - get best so far
            best = es.population[0] if hasattr(es, "population") and es.population else None
        finally:
            # Restore original ollama.chat if we monkey-patched it
            if ollama_module is not None and original_ollama_chat is not None:
                ollama_module.chat = original_ollama_chat

        wall_time = time.time() - start_time
        llm_time = stats["llm_time"]
        eval_count = stats["eval_count"]
        failure_count = stats["failure_count"]
        compute_time = wall_time - llm_time
        success_rate = 100.0 * (eval_count - failure_count) / eval_count if eval_count > 0 else 0.0

        # Extract fitness and code from Solution object
        best_fitness = float("inf")
        best_code = ""
        if best is not None:
            if hasattr(best, "fitness") and best.fitness is not None:
                best_fitness = best.fitness
            if hasattr(best, "code") and best.code is not None:
                best_code = str(best.code)

        # Calculate token totals and cost
        prompt_tokens = stats["prompt_tokens"]
        completion_tokens = stats["completion_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        # Estimate cost (OpenAI gpt-4o-mini pricing as of 2024)
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens)

        return LLaMEAResult(
            best_fitness=best_fitness,
            best_code=best_code,
            evaluations=eval_count,
            llm_calls=stats["llm_calls"],
            wall_time_seconds=wall_time,
            llm_time_seconds=llm_time,
            compute_time_seconds=compute_time,
            llm_failures=failure_count,
            llm_success_rate=success_rate,
            fitness_history=fitness_history,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            stopped_by_token_limit=stats["token_limit_reached"],
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on model pricing.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Pricing per 1M tokens (as of 2025)
        # Source: https://openai.com/api/pricing/
        pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5.1": {"input": 1.25, "output": 10.00},
            "gpt-5.2": {"input": 1.75, "output": 14.00},
            "gpt-5-mini": {"input": 0.30, "output": 1.25},
            "gpt-5-pro": {"input": 10.00, "output": 120.00},
        }

        # Default to gpt-4o-mini pricing for unknown models
        model_key = self.model if self.model in pricing else "gpt-4o-mini"
        rates = pricing[model_key]

        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost


def create_tsp_task_prompt(instance_name: str, n_cities: int) -> str:
    """Create task prompt for TSP optimization.

    Args:
        instance_name: Name of the TSP instance
        n_cities: Number of cities

    Returns:
        Task prompt string
    """
    return f"""You are designing a heuristic algorithm for the Traveling Salesman Problem (TSP).

Problem: {instance_name} with {n_cities} cities.

Your task is to write a Python function that takes a distance matrix and returns a tour.
The function should be named `solve_tsp` and have this signature:

def solve_tsp(distance_matrix: list[list[float]]) -> list[int]:
    '''
    Solve TSP instance.

    Args:
        distance_matrix: NxN matrix of distances

    Returns:
        Tour as list of city indices (0 to N-1)
    '''
    # Your implementation here
    pass

Requirements:
- Return a valid tour visiting all cities exactly once
- Minimize total tour length
- You can use standard Python libraries (random, math, etc.)
- Be efficient - the algorithm will be evaluated multiple times

Write only the function implementation, no additional code."""


def create_tsp_fitness_wrapper(
    distance_matrix: list[list[float]],
    timeout_seconds: int = 30,
) -> Callable[[str], float]:
    """Create fitness function that evaluates TSP solution code.

    Args:
        distance_matrix: TSP distance matrix
        timeout_seconds: Maximum time for code execution

    Returns:
        Function that evaluates code string
    """
    import signal

    n = len(distance_matrix)

    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timeout")

    def evaluate_code(code: str) -> float:
        """Evaluate TSP solution code.

        Args:
            code: Python code defining solve_tsp function

        Returns:
            Tour cost (lower is better), or infinity if invalid/timeout
        """
        try:
            # Execute code to define function
            # Include common modules in globals so LLM-generated code can use them
            import random
            import math
            import itertools
            import functools
            import collections

            exec_globals = {
                "__builtins__": __builtins__,
                "random": random,
                "math": math,
                "itertools": itertools,
                "functools": functools,
                "collections": collections,
            }
            local_vars: dict[str, Any] = {}
            exec(code, exec_globals, local_vars)

            if "solve_tsp" not in local_vars:
                return float("inf")

            solve_fn = local_vars["solve_tsp"]

            # Run solver with timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                tour = solve_fn(distance_matrix)
            finally:
                signal.alarm(0)  # Cancel the alarm

            # Validate tour
            if not isinstance(tour, list):
                return float("inf")
            if len(tour) != n:
                return float("inf")
            if set(tour) != set(range(n)):
                return float("inf")

            # Calculate tour cost
            total = 0.0
            for i in range(n):
                from_city = tour[i]
                to_city = tour[(i + 1) % n]
                total += distance_matrix[from_city][to_city]

            return total

        except (TimeoutError, Exception):
            return float("inf")

    return evaluate_code
