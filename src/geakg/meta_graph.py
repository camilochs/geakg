"""Meta-Graph: Domain-Agnostic Algorithm Composition Structure.

BACKWARD COMPATIBILITY STUB: This module re-exports from src.geakg.layers.l0.
New code should import directly from src.geakg.layers.l0:

    from src.geakg.layers.l0 import (
        MetaGraph, MetaEdge, InstantiatedGraph,
        create_ils_meta_graph, create_vns_meta_graph,
    )

This stub exists for backward compatibility with existing imports.
"""

# Core MetaGraph classes
from src.geakg.layers.l0.metagraph import (
    MetaGraph,
    MetaEdge,
    InstantiatedGraph,
)

# Factory functions
from src.geakg.layers.l0.patterns import (
    create_ils_meta_graph,
    create_vns_meta_graph,
    create_hybrid_meta_graph,
)

__all__ = [
    "MetaGraph",
    "MetaEdge",
    "InstantiatedGraph",
    "create_ils_meta_graph",
    "create_vns_meta_graph",
    "create_hybrid_meta_graph",
]
