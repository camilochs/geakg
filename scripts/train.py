#!/usr/bin/env python3
"""Unified Training CLI for NS-GE.

Abstrae los detalles de exp_pure_mode_tsp.py en una interfaz limpia.

Uso básico:
    # TSP training con synthesis synthesis
    uv run python scripts/train.py tsp --synthesis

    # Con instancias específicas
    uv run python scripts/train.py tsp \
        --instances berlin52,kroA100,ch150 \
        --synthesis

    # VRP con transfer desde TSP
    uv run python scripts/train.py vrp \
        --transfer-from experiments/nsgge/results/YYYYMMDD/akg_snapshot.json

Opciones comunes:
    --synthesis         Habilitar synthesis synthesis
    --model MODEL       Modelo LLM (default: gpt-4o-mini)
    --max-tokens N      Budget de tokens (default: 200000)
    --time-limit N      Límite de tiempo en segundos
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_tsp_instances(names: list[str] | None) -> list[str]:
    """Resolve TSP instance names to paths."""
    instance_dir = Path("data/instances/tsp")

    if not names:
        # Default instances
        names = ["berlin52", "kroA100", "ch150"]

    paths = []
    for name in names:
        # Handle full paths
        if "/" in name or name.endswith(".tsp"):
            if Path(name).exists():
                paths.append(name)
            continue

        # Search in instance directory
        candidates = list(instance_dir.glob(f"{name}*.tsp"))
        if candidates:
            paths.append(str(candidates[0]))
        else:
            print(f"Warning: Instance '{name}' not found in {instance_dir}")

    return paths


def get_vrp_instances(names: list[str] | None) -> list[str]:
    """Resolve VRP instance names to paths."""
    instance_dir = Path("data/instances/vrp")

    if not names:
        # Default instances
        names = ["A-n32-k5"]

    paths = []
    for name in names:
        if "/" in name or name.endswith(".vrp"):
            if Path(name).exists():
                paths.append(name)
            continue

        candidates = list(instance_dir.glob(f"{name}*.vrp"))
        if candidates:
            paths.append(str(candidates[0]))
        else:
            print(f"Warning: Instance '{name}' not found in {instance_dir}")

    return paths


def train_tsp(args):
    """Run TSP training."""
    from scripts.exp_pure_mode_tsp import run_multi_instance_experiment

    instances = get_tsp_instances(args.instances)
    if not instances:
        print("No instances found!")
        return 1

    print(f"Training TSP with {len(instances)} instances:")
    for inst in instances:
        print(f"  - {inst}")

    result = run_multi_instance_experiment(
        instance_files=instances,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        n_ants=args.ants,
        enable_synthesis=args.synthesis,
        immediate_synthesis=args.immediate_synthesis,
        async_synthesis=args.async_mode,
        use_openai=True,  # Always use OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=args.output_dir,
        save_techniques=args.save_techniques,
        load_techniques=args.load_techniques,
    )

    print(f"\nResults saved to: {result['session_dir']}")
    print(f"Final gap: {result['aggregate_gap']:.2f}%")

    return 0


def train_vrp(args):
    """Run VRP training with optional transfer from TSP."""
    if args.transfer_from:
        # Transfer learning mode
        from scripts.run_vrp_transfer import main as run_transfer
        sys.argv = [
            "run_vrp_transfer.py",
            "--snapshot", args.transfer_from,
            "--time-limit", str(args.time_limit or 60),
        ]
        if args.instances:
            instances = get_vrp_instances(args.instances)
            if instances:
                sys.argv.extend(["--instance", instances[0]])

        run_transfer()
    else:
        print("VRP training without transfer not yet implemented.")
        print("Use --transfer-from to load TSP operators.")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="NS-GE Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic TSP training with synthesis synthesis
  uv run python scripts/train.py tsp --synthesis

  # TSP with specific instances, model and token budget
  uv run python scripts/train.py tsp \\
      --instances berlin52,kroA100,ch150 \\
      --synthesis \\
      --model gpt-5.2 \\
      --max-tokens 200000

  # VRP with transfer learning from TSP
  uv run python scripts/train.py vrp \\
      --transfer-from experiments/nsgge/results/*/akg_snapshot.json
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="domain", help="Domain to train")

    # TSP subcommand
    tsp_parser = subparsers.add_parser("tsp", help="Train on TSP")
    tsp_parser.add_argument(
        "--instances", "-i",
        type=lambda x: [s.strip() for s in x.split(",")],
        help="Comma-separated instance names (e.g., berlin52,kroA100)",
    )
    tsp_parser.add_argument(
        "--synthesis", "-s",
        action="store_true",
        help="Enable synthesis synthesis",
    )
    tsp_parser.add_argument(
        "--immediate-synthesis",
        action="store_true",
        help="Start synthesis immediately (don't wait for stagnation)",
    )
    tsp_parser.add_argument(
        "--async-mode", "-a",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run synthesis asynchronously (default: True). Use --no-async-mode for sync.",
    )
    tsp_parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="LLM model (default: gpt-4o-mini)",
    )
    tsp_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="GPT-5 reasoning effort",
    )
    tsp_parser.add_argument(
        "--max-tokens",
        type=int,
        default=200000,
        help="Token budget (default: 200000)",
    )
    tsp_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Max runtime in seconds (ACO continues after max_tokens, synthesis stops)",
    )
    tsp_parser.add_argument(
        "--ants",
        type=int,
        default=15,
        help="Number of ants (default: 15)",
    )
    tsp_parser.add_argument(
        "--output-dir",
        default="experiments/nsgge/results",
        help="Output directory",
    )
    tsp_parser.add_argument(
        "--save-techniques",
        help="Save learned techniques to file",
    )
    tsp_parser.add_argument(
        "--load-techniques",
        help="Load techniques for transfer learning",
    )

    # VRP subcommand
    vrp_parser = subparsers.add_parser("vrp", help="Train on VRP")
    vrp_parser.add_argument(
        "--instances", "-i",
        type=lambda x: [s.strip() for s in x.split(",")],
        help="Comma-separated instance names",
    )
    vrp_parser.add_argument(
        "--transfer-from", "-t",
        help="Path to TSP snapshot for transfer learning",
    )
    vrp_parser.add_argument(
        "--time-limit",
        type=float,
        default=60,
        help="Time limit in seconds (default: 60)",
    )

    args = parser.parse_args()

    if not args.domain:
        parser.print_help()
        return 1

    if args.domain == "tsp":
        return train_tsp(args)
    elif args.domain == "vrp":
        return train_vrp(args)
    else:
        print(f"Unknown domain: {args.domain}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
