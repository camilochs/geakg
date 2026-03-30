"""GEAKG Layers: 3-layer architecture for NS-SE.

Architecture:
    L0: MetaGraph Topology (offline, LLM)
        - Abstract roles (11 semantic behaviors)
        - MetaGraph topology (transitions between roles)
        - Initial weights (LLM heuristic preferences)
        - Conditions (adaptive control policies)

    L1: Operator Generation (offline, LLM)
        - Operator pool (executable code per role)
        - AFO generation (Always-From-Original)
        - Design-space prompting

    L2: ACO Training (offline, ACO)
        - Pheromones (learned weights)
        - Symbolic rules (patterns for online execution)
        - Snapshots (serialized GEAKG state)

Usage:
    # L0: Topology
    from src.geakg.layers.l0 import MetaGraph, AbstractRole, EdgeCondition

    # L1: Operators
    from src.geakg.layers.l1 import OperatorPool, L1Generator

    # L2: Learned knowledge
    from src.geakg.layers.l2 import GEAKGSnapshot, PheromoneMatrix
"""

# Re-export main classes for convenience
from src.geakg.layers.l0 import (
    AbstractRole,
    RoleCategory,
    MetaGraph,
    MetaEdge,
    EdgeCondition,
    ConditionType,
)

from src.geakg.layers.l1 import (
    Operator,
    OperatorPool,
    L1Generator,
    L1Config,
)

from src.geakg.layers.l2 import (
    GEAKGSnapshot,
    PheromoneMatrix,
    SymbolicRule,
)

__all__ = [
    # L0
    "AbstractRole",
    "RoleCategory",
    "MetaGraph",
    "MetaEdge",
    "EdgeCondition",
    "ConditionType",
    # L1
    "Operator",
    "OperatorPool",
    "L1Generator",
    "L1Config",
    # L2
    "GEAKGSnapshot",
    "PheromoneMatrix",
    "SymbolicRule",
]
