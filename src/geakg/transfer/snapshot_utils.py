"""Snapshot utilities for transfer learning.

Functions to find and load AKG snapshots for transfer.
"""

import json
from pathlib import Path


def find_latest_snapshot_with_operators(
    experiments_dir: str = "experiments/nsgge/results",
    test_snapshot: str = "/tmp/test_tsp_snapshot.json",
) -> str | None:
    """Find most recent AKG snapshot with synthesized operators.

    Searches for snapshots in experiments directory and returns
    the most recently modified one that contains synthesized synthesized operators.

    Args:
        experiments_dir: Base directory for experiment results
        test_snapshot: Path to check for test snapshot first

    Returns:
        Path to snapshot file, or None if not found
    """
    # First try test snapshot
    test_path = Path(test_snapshot)
    if test_path.exists():
        try:
            with open(test_path) as f:
                data = json.load(f)
            # Check for synthesized operators in either format
            if data.get("synth_operators") or data.get("operators", {}).get("synthesized_synth"):
                return str(test_path)
        except (json.JSONDecodeError, KeyError):
            pass

    # Search experiments directory
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        return None

    candidates = []
    for run_dir in exp_path.iterdir():
        if not run_dir.is_dir():
            continue

        snapshot = run_dir / "akg_snapshot.json"
        if not snapshot.exists():
            continue

        try:
            with open(snapshot) as f:
                data = json.load(f)

            # Check for synthesized operators in either format
            has_synth = (
                data.get("synth_operators") or
                data.get("operators", {}).get("synthesized_synth")
            )

            if has_synth:
                candidates.append((snapshot.stat().st_mtime, snapshot))
        except (json.JSONDecodeError, KeyError):
            continue

    if not candidates:
        return None

    # Return most recent
    candidates.sort(reverse=True)
    return str(candidates[0][1])


def get_synth_operators_from_snapshot(snapshot: dict) -> list[dict]:
    """Extract synthesized operators from a snapshot dict.

    Handles both old and new snapshot formats.

    Args:
        snapshot: Loaded snapshot dict

    Returns:
        List of synthesized operator dicts
    """
    # New format: operators.synthesized_synth
    synth_ops = snapshot.get("operators", {}).get("synthesized_synth", [])
    if synth_ops:
        return synth_ops

    # Old format: synth_operators dict
    old_format = snapshot.get("synth_operators", {})
    if old_format:
        # Convert dict to list format
        return [
            {"operator_id": op_id, **op_data}
            for op_id, op_data in old_format.items()
        ]

    return []


def get_pheromones_from_snapshot(snapshot: dict) -> dict:
    """Extract pheromone data from a snapshot.

    Args:
        snapshot: Loaded snapshot dict

    Returns:
        Pheromones dict with role_level key
    """
    return snapshot.get("pheromones", {})
