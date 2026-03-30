#!/usr/bin/env python3
"""Run iterative pool refinement: ACO-guided operator generation.

This script implements the iterative refinement loop:
1. Start with generic operators (base pool)
2. Run ACO to discover structural patterns
3. Analyze snapshot to find "weak spots"
4. Generate operators specifically for those weak contexts
5. Repeat until pool is good enough
6. Save snapshot for transfer to similar problems

Usage:
    # Basic usage with default pool
    uv run python scripts/run_iterative_refinement.py \
        --instances-dir data/instances/tsp_generated \
        --n-instances 5 \
        --output-dir experiments/iterative

    # With existing pool
    uv run python scripts/run_iterative_refinement.py \
        --pool pools/tsp_pool.json \
        --instances-dir data/instances/tsp_generated \
        --max-rounds 3

    # Full configuration
    uv run python scripts/run_iterative_refinement.py \
        --instances-dir data/instances/tsp_generated \
        --n-instances 10 \
        --max-rounds 5 \
        --aco-timeout 120 \
        --model gpt-4o-mini \
        --output-dir experiments/iterative
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


def create_base_pool():
    """Create a base pool with generic operators."""
    from src.geakg.layers.l1.pool import Operator, OperatorPool
    from src.geakg.layers.l1.base_operators import BASE_OPERATORS

    pool = OperatorPool(
        metadata={
            "domain": "tsp",
            "type": "base_generic",
            "description": "Generic operators for iterative refinement",
        }
    )

    # Add base operators from each role
    for role, code in BASE_OPERATORS.items():
        op = Operator(
            name=f"{role}_base",
            code=code.strip(),
            role=role,
            design_choices={"type": "generic"},
            interaction_effects="Base generic operator",
        )
        pool.add_operator(op)

    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative pool refinement with ACO-guided generation"
    )

    # Instance selection (use --instances for specific files, or --instances-dir for directory)
    parser.add_argument("--instances", nargs="+", help="Specific instance files (e.g., tsp50_001.tsp tsp100_002.tsp)")
    parser.add_argument("--instances-dir", help="Directory with TSP instances")
    parser.add_argument("--n-instances", type=int, default=5, help="Number of instances (only with --instances-dir)")

    # Pool options
    parser.add_argument("--pool", help="Path to existing pool (default: create base pool)")

    # Refinement configuration
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum refinement rounds")
    parser.add_argument("--aco-timeout", type=int, default=60, help="ACO timeout per round (seconds)")
    parser.add_argument("--weak-spots-per-round", type=int, default=3, help="Weak spots to address per round")
    parser.add_argument("--n-ants", type=int, default=15, help="Number of ants per ACO iteration")

    # LLM options
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for generation")
    parser.add_argument("--backend", default="openai", choices=["openai", "ollama"],
                        help="LLM backend: openai or ollama (default: openai)")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for GPT-5 models (default: none for gpt-5.1/5.2, medium for gpt-5)")
    parser.add_argument("--max-tokens", type=int, help="Max total tokens (stops training when reached)")
    parser.add_argument(
        "--use-llm-metagraph",
        action="store_true",
        default=True,
        help="Use LLM to generate metagraph topology (L2/L3). Default: True",
    )
    parser.add_argument(
        "--no-llm-metagraph",
        action="store_true",
        help="Use predefined metagraph instead of LLM-generated",
    )

    # LLaMEA hybrid mode (live invocation)
    parser.add_argument(
        "--llamea-hybrid",
        action="store_true",
        help="Use LLaMEA live for ls_intensify_large (invokes LLaMEA during execution)",
    )
    parser.add_argument(
        "--llamea-budget",
        type=int,
        default=5000,
        help="Token budget for each LLaMEA invocation in hybrid mode (default: 5000)",
    )

    # Output
    parser.add_argument("--output-dir", default="experiments/iterative", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    random.seed(args.seed)

    # Setup session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / f"{timestamp}_iterative"
    session_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(session_dir / "run.log", "w")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 60)
    log("ITERATIVE POOL REFINEMENT")
    log("=" * 60)

    # Imports
    from src.geakg.instance_pool import InstancePool
    from src.geakg.layers.l1 import OperatorPool
    from src.geakg.offline.iterative_refinement import (
        IterativeRefinementConfig,
        run_iterative_refinement,
        save_snapshot_for_transfer,
    )
    from src.domains import get_domain_config
    from src.llm.client import OllamaClient, OpenAIClient
    from src.llm.config import LLMConfig

    # Load instances
    log("\n--- Loading Instances ---")
    instance_pool = InstancePool(domain="tsp")

    if args.instances:
        # Load specific instance files
        instance_pool.load_instances_from_files(args.instances)
    elif args.instances_dir:
        # Load from directory
        instance_pool.load_instances_from_dir(args.instances_dir, limit=args.n_instances)
    else:
        log("ERROR: Must specify --instances or --instances-dir")
        sys.exit(1)

    if len(instance_pool) == 0:
        log("ERROR: No instances found")
        sys.exit(1)

    log(f"Loaded {len(instance_pool)} instances:")
    for inst in instance_pool.instances:
        opt_str = f" (opt={inst.optimal})" if inst.optimal else ""
        log(f"  - {inst.instance_id}: {inst.dimension} cities{opt_str}")

    # Load or create pool
    log("\n--- Operator Pool ---")
    if args.pool:
        pool = OperatorPool.load(args.pool)
        log(f"Loaded pool from {args.pool}: {pool.total_operators} operators")
    else:
        pool = create_base_pool()
        log(f"Created base pool: {pool.total_operators} operators")

    for role in pool.roles:
        ops = pool.get_operators_for_role(role)
        log(f"  {role}: {len(ops)} operators")

    # LLaMEA hybrid mode: create live wrapper that invokes LLaMEA in real-time
    llamea_wrapper = None
    if args.llamea_hybrid:
        log("\n--- LLaMEA Hybrid Mode (Live) ---")

        from src.geakg.operators.llamea_live_wrapper import LLaMEALiveWrapper

        # Build training instances for multi-instance fitness evaluation
        # This ensures LLaMEA evolves code that generalizes across all instances
        training_instances = [
            {
                "name": inst.instance_id,
                "dimension": inst.dimension,
                "distance_matrix": inst.instance_data["distance_matrix"],
                "optimal": inst.optimal,
            }
            for inst in instance_pool.instances
        ]

        # Create live wrapper that will invoke LLaMEA during execution
        llamea_wrapper = LLaMEALiveWrapper(
            name="llamea_live_ls",
            role="ls_intensify_large",
            model=args.model,
            max_tokens=args.llamea_budget,
            llm_backend=args.backend,
            api_key=os.getenv("OPENAI_API_KEY"),
            eval_timeout=10,
            training_instances=training_instances,
        )
        log(f"LLaMEA live wrapper created:")
        log(f"  Model: {args.model}")
        log(f"  Token budget: {args.llamea_budget}")
        log(f"  Role: ls_intensify_large")
        log(f"  Training instances: {len(training_instances)} (multi-instance fitness)")

        # Remove base operator from ls_intensify_large to avoid O(n^4) bottleneck
        # LLaMEA will be the only operator for this role
        base_op_name = "ls_intensify_large_base"
        if pool.remove_operator(base_op_name):
            log(f"  Removed '{base_op_name}' to avoid O(n^4) bottleneck")

    # Setup domain config
    domain_config = get_domain_config("tsp")

    # Setup LLM client (cache disabled for reproducible experiments)
    log("\n--- LLM Client ---")
    if args.backend == "ollama":
        config = LLMConfig(
            model=args.model,
            cache_enabled=False,
        )
        llm_client = OllamaClient(config=config)
        log(f"Backend: Ollama (local)")
    else:
        llm_client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=args.model,
            cache_enabled=False,
            reasoning_effort=args.reasoning_effort,
        )
        log(f"Backend: OpenAI")
    log(f"Model: {args.model} (cache disabled)")
    if args.reasoning_effort:
        log(f"Reasoning effort: {args.reasoning_effort}")

    # Context factory for validation
    def ctx_factory(instance_data):
        return domain_config.create_context(instance_data)

    # Refinement config
    config = IterativeRefinementConfig(
        max_rounds=args.max_rounds,
        aco_timeout=args.aco_timeout,
        weak_spots_per_round=args.weak_spots_per_round,
        n_ants=args.n_ants,
    )

    # Determine metagraph mode
    use_llm_metagraph = not args.no_llm_metagraph

    log("\n--- Configuration ---")
    log(f"Max rounds: {config.max_rounds}")
    log(f"ACO timeout: {config.aco_timeout}s")
    log(f"Weak spots per round: {config.weak_spots_per_round}")
    log(f"Number of ants: {config.n_ants}")
    log(f"Max tokens: {args.max_tokens or 'unlimited'}")
    log(f"LLM metagraph: {use_llm_metagraph}")
    log(f"LLaMEA hybrid: {args.llamea_hybrid}")

    # Run iterative refinement
    log("\n" + "=" * 60)
    log("STARTING ITERATIVE REFINEMENT")
    log("=" * 60)

    start_time = time.time()

    refined_pool, final_snapshot = run_iterative_refinement(
        pool=pool,
        instance_pool=instance_pool,
        domain_config=domain_config,
        llm_client=llm_client,
        config=config,
        ctx_factory=ctx_factory,
        use_llm_metagraph=use_llm_metagraph,
        output_dir=session_dir,
        max_tokens=args.max_tokens,
        llamea_wrapper=llamea_wrapper,
    )

    elapsed = time.time() - start_time

    # Save results
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    log(f"Total time: {elapsed:.1f}s")
    log(f"Final pool: {refined_pool.total_operators} operators")
    log(f"Best gap: {final_snapshot.get('best_gap', 'N/A'):.2f}%")
    log(f"Refinement rounds: {final_snapshot.get('refinement_rounds', 0)}")

    # Show history
    log("\n--- Refinement History ---")
    for entry in final_snapshot.get("history", []):
        log(f"  Round {entry['round']}: gap={entry['gap']:.2f}%, weak_spots={entry['weak_spots']}")

    # Save pool
    pool_path = session_dir / "refined_pool.json"
    refined_pool.save(str(pool_path))
    log(f"\nSaved pool: {pool_path}")

    # Save snapshot for transfer
    snapshot_path = session_dir / "akg_snapshot.json"
    save_snapshot_for_transfer(final_snapshot, str(snapshot_path))
    log(f"Saved snapshot: {snapshot_path}")

    # Save run config
    config_path = session_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "instances_dir": args.instances_dir,
            "n_instances": args.n_instances,
            "pool": args.pool,
            "max_rounds": args.max_rounds,
            "aco_timeout": args.aco_timeout,
            "weak_spots_per_round": args.weak_spots_per_round,
            "n_ants": args.n_ants,
            "model": args.model,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "llamea_hybrid": args.llamea_hybrid,
            "llamea_budget": args.llamea_budget if args.llamea_hybrid else None,
        }, f, indent=2)

    log_file.close()

    print(f"\n{'=' * 60}")
    print("ITERATIVE REFINEMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Pool: {refined_pool.total_operators} operators")
    print(f"Best gap: {final_snapshot.get('best_gap', 'N/A'):.2f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"Session: {session_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
