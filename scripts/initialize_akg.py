#!/usr/bin/env python3
"""Initialize and save the default Algorithmic Knowledge Graph."""

from pathlib import Path

from src.geakg import create_default_akg, get_operator_summary


def main():
    """Create and save the default AKG."""
    print("=" * 60)
    print("Initializing Algorithmic Knowledge Graph (AKG)")
    print("=" * 60)

    # Create AKG
    akg = create_default_akg()

    print(f"\nCreated AKG: {akg}")
    print(f"  Total nodes: {len(akg.nodes)}")
    print(f"  Total edges: {len(akg.edges)}")

    # Print operator summary
    summary = get_operator_summary()
    print("\nOperators by category:")
    for category, operators in summary.items():
        print(f"  {category}: {len(operators)} operators")
        for op in operators[:3]:
            print(f"    - {op}")
        if len(operators) > 3:
            print(f"    ... and {len(operators) - 3} more")

    # Save to GraphML
    output_dir = Path(__file__).parent.parent / "data" / "akg_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "default_akg.graphml"
    akg.to_graphml(output_path)
    print(f"\nSaved AKG to: {output_path}")

    # Verify by loading
    from src.geakg.graph import AlgorithmicKnowledgeGraph

    loaded_akg = AlgorithmicKnowledgeGraph.from_graphml(output_path)
    print(f"Verified loading: {loaded_akg}")

    # Show some valid transitions
    print("\nSample valid transitions:")
    start_ops = ["greedy_nearest_neighbor", "two_opt", "double_bridge"]
    for op in start_ops:
        transitions = akg.get_valid_transitions(op)
        print(f"  {op} -> {transitions[:3]}")

    print("\n" + "=" * 60)
    print("AKG initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
