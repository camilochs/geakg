#!/usr/bin/env python3
"""Generate L0 operator pool using AFO + Design-Space Prompting.

This script generates an offline pool of operators for a specific domain.
The pool can then be used at runtime without any LLM calls.

Usage:
    # Generate pool for TSP with 50k token budget
    uv run python scripts/generate_l0_pool.py \
        --domain tsp \
        --instances data/instances/tsp_generated/tsp50_*.tsp \
        --max-tokens 50000 \
        --output pools/tsp_pool.json

    # Generate with custom settings
    uv run python scripts/generate_l0_pool.py \
        --domain tsp \
        --instances data/instances/tsp/berlin52.tsp \
        --max-tokens 100000 \
        --pool-size 5 \
        --racing-instances 10 \
        --seed 42 \
        --output pools/tsp_custom.json
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env file if exists
load_dotenv()


def create_tsp_evaluate_fn():
    """Create evaluation function for TSP operators."""
    from src.geakg.contexts.tsp import TSPContext

    def evaluate_fn(operator, instance):
        """Evaluate operator on TSP instance.

        Args:
            operator: Operator with code attribute
            instance: Dict with 'distance_matrix' and 'dimension'

        Returns:
            Tour cost (lower is better)
        """
        # Compile operator
        namespace = {}
        exec(compile(operator.code, "<string>", "exec"), namespace)

        # Find function
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            return float("inf")

        # Create context and initial solution
        ctx = TSPContext(instance["distance_matrix"])
        n = instance["dimension"]
        initial_solution = list(range(n))

        try:
            # Run operator
            result = func(initial_solution, ctx)

            # Validate and evaluate
            if not isinstance(result, list) or len(result) != n:
                return float("inf")
            if set(result) != set(range(n)):
                return float("inf")

            return ctx.evaluate(result)
        except Exception:
            return float("inf")

    return evaluate_fn


def create_tsp_ctx_factory():
    """Create context factory for TSP."""
    from src.geakg.contexts.tsp import TSPContext

    def ctx_factory(instance):
        return TSPContext(instance["distance_matrix"])

    return ctx_factory


def load_tsp_instances(patterns: list[str]) -> list[dict]:
    """Load TSP instances from file patterns.

    Args:
        patterns: List of glob patterns or file paths

    Returns:
        List of instance dictionaries
    """
    from src.domains.tsp import TSPDomain

    domain = TSPDomain()
    instances = []

    for pattern in patterns:
        paths = glob.glob(pattern)
        if not paths:
            # Try as direct path
            if Path(pattern).exists():
                paths = [pattern]

        for path in paths:
            try:
                tsp_instance = domain.load_instance(Path(path))
                instances.append({
                    "name": tsp_instance.name,
                    "dimension": tsp_instance.dimension,
                    "distance_matrix": tsp_instance.distance_matrix,
                    "optimal_cost": tsp_instance.optimal_cost,
                })
                logger.info(f"Loaded {tsp_instance.name} (n={tsp_instance.dimension})")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    return instances


def main():
    parser = argparse.ArgumentParser(
        description="Generate L0 operator pool using AFO + Design-Space Prompting"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="tsp",
        choices=["tsp"],  # Add more domains as implemented
        help="Target domain",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        required=True,
        help="Instance file paths or glob patterns",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
        help="Maximum tokens to spend on generation",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5,
        help="Number of operators to keep per role after racing",
    )
    parser.add_argument(
        "--racing-instances",
        type=int,
        default=10,
        help="Number of instances for F-Race selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the pool JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (e.g., gpt-4o-mini, gpt-4o, qwen2.5:14b)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "ollama"],
        help="LLM provider",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    logger.info(f"L0 Pool Generation for {args.domain}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Pool size per role: {args.pool_size}")

    # Load instances
    logger.info("Loading instances...")
    if args.domain == "tsp":
        instances = load_tsp_instances(args.instances)
        evaluate_fn = create_tsp_evaluate_fn()
        ctx_factory = create_tsp_ctx_factory()
    else:
        logger.error(f"Unsupported domain: {args.domain}")
        sys.exit(1)

    if not instances:
        logger.error("No instances loaded!")
        sys.exit(1)

    logger.info(f"Loaded {len(instances)} instances")

    # Create LLM client
    logger.info(f"Initializing {args.provider} client with model {args.model}")
    if args.provider == "openai":
        from src.llm.client import OpenAIClient

        llm_client = OpenAIClient(model=args.model, api_key=args.api_key)
    else:
        from src.llm.client import OllamaClient
        from src.llm.config import LLMConfig

        llm_client = OllamaClient(config=LLMConfig(model=args.model))

    # Create generator
    from src.geakg.layers.l1 import L1Config as L0Config, L1Generator as L0Generator

    config = L0Config(
        max_tokens=args.max_tokens,
        pool_size_per_role=args.pool_size,
        racing_instances=args.racing_instances,
        temperature=args.temperature,
        seed=args.seed,
    )

    generator = L0Generator(
        llm_client=llm_client,
        domain=args.domain,
        config=config,
    )

    # Generate pool
    logger.info("Starting generation...")
    pool = generator.generate(
        instances=instances,
        evaluate_fn=evaluate_fn,
        ctx_factory=ctx_factory,
    )

    # Save pool
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool.save(output_path)

    logger.info(f"Pool saved to {output_path}")
    logger.info(f"Total operators: {pool.total_operators}")
    logger.info(f"Roles: {pool.roles}")

    # Print summary
    print("\n" + "=" * 60)
    print("L0 POOL GENERATION COMPLETE")
    print("=" * 60)
    print(f"Domain: {args.domain}")
    print(f"Tokens used: {generator.tokens_used}")
    print(f"Total operators: {pool.total_operators}")
    print(f"Output: {output_path}")
    print("\nOperators per role:")
    for role in sorted(pool.roles):
        ops = pool.get_operators_for_role(role)
        best = pool.get_best_for_role(role)
        best_fitness = f"{best.avg_fitness:.2f}" if best and best.fitness_scores else "N/A"
        print(f"  {role}: {len(ops)} operators (best fitness: {best_fitness})")
    print("=" * 60)


if __name__ == "__main__":
    main()
