"""Offline Phase: Training pipeline for NS-SE.

This module contains the training components that use LLM (L0, L1)
and ACO (L2) to build the GEAKG:

- aco_trainer: ACO-based training to learn pheromones
- iterative_refinement: Multi-round training loop
- persistence: Save/load training artifacts

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/offline/  <-- YOU ARE HERE
    Online: Symbolic Executor - src/geakg/online/

Usage:
    from src.geakg.offline import (
        run_iterative_refinement,
        ACOTrainer,
        save_snapshot_for_transfer,
    )
"""

from src.geakg.offline.iterative_refinement import (
    WeakSpot,
    SnapshotAnalysis,
    IterativeRefinementConfig,
    analyze_snapshot,
    generate_contextual_operator,
    run_iterative_refinement,
    save_snapshot_for_transfer,
    load_snapshot_for_transfer,
)

__all__ = [
    # Iterative refinement
    "WeakSpot",
    "SnapshotAnalysis",
    "IterativeRefinementConfig",
    "analyze_snapshot",
    "generate_contextual_operator",
    "run_iterative_refinement",
    "save_snapshot_for_transfer",
    "load_snapshot_for_transfer",
]
