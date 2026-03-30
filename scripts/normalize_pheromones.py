#!/usr/bin/env python3
"""Normalize pheromone values in AKG snapshots from [0.01, 10] to [0.01, 1.0].

This script updates all akg_snapshot.json files to use the standard MMAS
pheromone range of [0.01, 1.0] instead of the legacy [0.01, 10.0] range.
"""

import json
from pathlib import Path


def normalize_pheromones(data: dict, old_max: float = 10.0, new_max: float = 1.0) -> dict:
    """Normalize pheromone values in a snapshot dictionary."""
    scale = new_max / old_max

    def normalize_dict(d: dict) -> dict:
        """Normalize all float values in a pheromone dictionary."""
        return {k: round(v * scale, 6) for k, v in d.items()}

    # Normalize top-level pheromones
    if 'pheromones' in data:
        pheromones = data['pheromones']
        if 'role_level' in pheromones:
            pheromones['role_level'] = normalize_dict(pheromones['role_level'])
        if 'operator_level' in pheromones:
            pheromones['operator_level'] = normalize_dict(pheromones['operator_level'])

    # Normalize pheromones in history entries
    if 'history' in data:
        for entry in data['history']:
            if 'pheromones' in entry:
                pheromones = entry['pheromones']
                if 'role_level' in pheromones:
                    pheromones['role_level'] = normalize_dict(pheromones['role_level'])
                if 'operator_level' in pheromones:
                    pheromones['operator_level'] = normalize_dict(pheromones['operator_level'])

    return data


def process_snapshot(path: Path, dry_run: bool = False) -> bool:
    """Process a single snapshot file.

    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(path) as f:
            data = json.load(f)

        # Check if already normalized (max value <= 1.0)
        max_val = 0.0
        if 'pheromones' in data:
            for level in ['role_level', 'operator_level']:
                if level in data['pheromones']:
                    vals = data['pheromones'][level].values()
                    if vals:
                        max_val = max(max_val, max(vals))

        if max_val <= 1.0:
            print(f"  SKIP (already normalized): {path}")
            return False

        # Normalize
        normalized = normalize_pheromones(data)

        if dry_run:
            print(f"  WOULD NORMALIZE: {path} (max={max_val:.2f})")
        else:
            with open(path, 'w') as f:
                json.dump(normalized, f, indent=2)
            print(f"  NORMALIZED: {path} (max={max_val:.2f} -> 1.0)")

        return True

    except Exception as e:
        print(f"  ERROR: {path}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Normalize pheromones in AKG snapshots')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--path', type=Path, default=Path('.'), help='Root path to search for snapshots')
    args = parser.parse_args()

    print(f"Searching for akg_snapshot.json files in {args.path.absolute()}...")

    snapshots = list(args.path.rglob('akg_snapshot.json'))
    print(f"Found {len(snapshots)} snapshot files\n")

    modified = 0
    skipped = 0
    errors = 0

    for path in sorted(snapshots):
        result = process_snapshot(path, dry_run=args.dry_run)
        if result:
            modified += 1
        elif result is False:
            skipped += 1
        else:
            errors += 1

    print(f"\nSummary:")
    print(f"  Modified: {modified}")
    print(f"  Skipped (already normalized): {skipped}")
    print(f"  Errors: {errors}")

    if args.dry_run:
        print("\n(Dry run - no files were actually modified)")


if __name__ == '__main__':
    main()
