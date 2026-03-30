#!/usr/bin/env python3
"""Migrate AKG snapshots to GEAKG format with L0/L1/L2 structure.

This script converts old akg_snapshot.json files to the new geakg_snapshot.json
format with explicit L0/L1/L2 layer separation.

Changes:
- Rename akg_snapshot.json → geakg_snapshot.json
- Restructure content to use l0_topology, l1_operators, l2_pheromones
- Update version to "2.0"

Usage:
    python scripts/migrate_snapshots.py --knowledge-dir experiments/nsse-llamea/knowledge
    python scripts/migrate_snapshots.py --snapshot path/to/akg_snapshot.json
"""

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger


def migrate_snapshot(old_path: Path, new_path: Path | None = None) -> Path:
    """Migrate a single snapshot to new GEAKG format.

    Args:
        old_path: Path to old akg_snapshot.json
        new_path: Optional new path (default: rename to geakg_snapshot.json)

    Returns:
        Path to migrated snapshot
    """
    if new_path is None:
        new_path = old_path.parent / "geakg_snapshot.json"

    # Read old snapshot
    with open(old_path) as f:
        old_data = json.load(f)

    # Create new structure
    new_data = {
        "name": old_data.get("name", old_path.parent.name),
        "domain": old_data.get("domain", "tsp"),
        "version": "2.0",

        # L0: Topology (from metagraph)
        "l0_topology": old_data.get("metagraph", {}),

        # L1: Operators (from operators_by_role)
        "l1_pool_ref": None,  # Will be set if pool file exists
        "l1_operators_by_role": old_data.get("operators_by_role", {}),

        # L2: Learned knowledge (from pheromones)
        "l2_pheromones": old_data.get("pheromones", {}),
        "l2_symbolic_rules": [],  # New field

        # Metadata
        "metadata": {
            **old_data.get("metadata", {}),
            "migrated_from": str(old_path),
            "original_version": old_data.get("version", "1.0"),
        },
    }

    # Check if pool file exists
    pool_path = old_path.parent / "refined_pool.json"
    if pool_path.exists():
        new_data["l1_pool_ref"] = str(pool_path)

    # Preserve any additional fields
    for key in old_data:
        if key not in ["name", "domain", "version", "metagraph", "operators_by_role",
                       "pheromones", "metadata"]:
            new_data["metadata"][f"legacy_{key}"] = old_data[key]

    # Write new snapshot
    with open(new_path, "w") as f:
        json.dump(new_data, f, indent=2)

    logger.info(f"Migrated {old_path} → {new_path}")
    return new_path


def migrate_directory(knowledge_dir: Path, backup: bool = True) -> list[Path]:
    """Migrate all snapshots in a knowledge directory.

    Args:
        knowledge_dir: Root directory containing snapshot subdirectories
        backup: Whether to backup original files

    Returns:
        List of migrated snapshot paths
    """
    migrated = []

    # Find all akg_snapshot.json files
    for snapshot_path in knowledge_dir.rglob("akg_snapshot.json"):
        if backup:
            backup_path = snapshot_path.with_suffix(".json.bak")
            shutil.copy(snapshot_path, backup_path)
            logger.info(f"Backed up {snapshot_path} → {backup_path}")

        new_path = migrate_snapshot(snapshot_path)
        migrated.append(new_path)

    # Also look for any geakg_snapshot.json that might need re-migration
    # (in case of format changes)
    for snapshot_path in knowledge_dir.rglob("geakg_snapshot.json"):
        with open(snapshot_path) as f:
            data = json.load(f)

        # Check if it's already v2.0
        if data.get("version") == "2.0" and "l0_topology" in data:
            logger.info(f"Skipping {snapshot_path} (already v2.0)")
            continue

        if backup:
            backup_path = snapshot_path.with_suffix(".json.bak")
            shutil.copy(snapshot_path, backup_path)

        # Re-migrate
        new_path = migrate_snapshot(snapshot_path, snapshot_path)
        migrated.append(new_path)

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate AKG snapshots to GEAKG format")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        help="Directory containing knowledge snapshots to migrate",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Single snapshot file to migrate",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for single snapshot migration",
    )

    args = parser.parse_args()

    if args.snapshot:
        # Migrate single file
        migrate_snapshot(args.snapshot, args.output)
    elif args.knowledge_dir:
        # Migrate directory
        migrated = migrate_directory(args.knowledge_dir, backup=not args.no_backup)
        print(f"\nMigrated {len(migrated)} snapshots")
    else:
        parser.print_help()
        return

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
