"""L2: ACO Training Layer for NS-SE.

This layer learns empirical knowledge through ACO training:
- Pheromones: learned weights that refine L0 initial weights
- Symbolic rules: patterns extracted from successful paths (for online execution)
- Snapshots: serialized GEAKG state (L0 topology + L1 pool + L2 learned)

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/  <-- YOU ARE HERE
    Online: Symbolic Executor - src/geakg/online/

Usage:
    from src.geakg.layers.l2 import (
        PheromoneMatrix,
        GEAKGSnapshot,
        extract_symbolic_rules,
    )
"""

from src.geakg.layers.l2.pheromones import PheromoneMatrix, PheromoneEntry
from src.geakg.layers.l2.snapshot import GEAKGSnapshot
from src.geakg.layers.l2.symbolic_rules import SymbolicRule, extract_symbolic_rules

__all__ = [
    # Pheromones
    "PheromoneMatrix",
    "PheromoneEntry",
    # Snapshot
    "GEAKGSnapshot",
    # Symbolic rules
    "SymbolicRule",
    "extract_symbolic_rules",
]
