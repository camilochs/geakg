"""NAS-Bench-Graph Architecture representation.

NAS-Bench-Graph uses a GNN cell search space:
- DAG with 6 nodes: 1 input + 4 computing nodes + 1 output
- Each computing node has:
  - 1 connectivity index in [0,3]: which prior node feeds input
  - 1 operation from 9 GNN options
- Operations: {gcn, gat, sage, gin, cheb, arma, k-gnn, identity, fc}
- 26,206 unique architectures (after isomorphism deduplication)
- Metric: accuracy (higher is better)

Reference: Qin et al., "NAS-Bench-Graph: Benchmarking Graph Neural
Architecture Search", NeurIPS 2022 Datasets and Benchmarks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# NAS-Bench-Graph operation vocabulary (order matters for encoding)
GRAPH_OPERATIONS = [
    "gcn",       # 0: Graph Convolutional Network
    "gat",       # 1: Graph Attention Network
    "sage",      # 2: GraphSAGE
    "gin",       # 3: Graph Isomorphism Network
    "cheb",      # 4: Chebyshev spectral convolution
    "arma",      # 5: ARMA filter
    "k-gnn",     # 6: k-dimensional GNN
    "identity",  # 7: identity / skip connection
    "fc",        # 8: fully connected layer
]

GRAPH_NUM_NODES = 4       # Number of computing nodes
GRAPH_NUM_OPS = len(GRAPH_OPERATIONS)  # 9 operations
GRAPH_MAX_CONN = 4        # Connectivity range [0, 3]


@dataclass
class GraphArchitecture:
    """A GNN architecture in the NAS-Bench-Graph search space.

    Represents a DAG with 4 computing nodes, each assigned:
    - connectivity[i]: int in [0, 3] — which prior node feeds this node
      (0 = input node, 1-3 = computing node index)
    - operations[i]: int in [0, 8] — GNN operation index

    Encoding: 8 integers total (4 connectivity + 4 operations).
    """

    connectivity: list[int] = field(
        default_factory=lambda: [0] * GRAPH_NUM_NODES
    )
    operations: list[int] = field(
        default_factory=lambda: [0] * GRAPH_NUM_NODES
    )

    def __post_init__(self) -> None:
        if len(self.connectivity) != GRAPH_NUM_NODES:
            raise ValueError(
                f"Expected {GRAPH_NUM_NODES} connectivity values, "
                f"got {len(self.connectivity)}"
            )
        if len(self.operations) != GRAPH_NUM_NODES:
            raise ValueError(
                f"Expected {GRAPH_NUM_NODES} operations, "
                f"got {len(self.operations)}"
            )
        for i, c in enumerate(self.connectivity):
            if not (0 <= c < GRAPH_MAX_CONN):
                raise ValueError(
                    f"Connectivity {i} has value {c}, "
                    f"must be in [0, {GRAPH_MAX_CONN - 1}]"
                )
        for i, op in enumerate(self.operations):
            if not (0 <= op < GRAPH_NUM_OPS):
                raise ValueError(
                    f"Operation {i} has value {op}, "
                    f"must be in [0, {GRAPH_NUM_OPS - 1}]"
                )

    @staticmethod
    def random(rng: random.Random | None = None) -> GraphArchitecture:
        """Generate a random GNN architecture.

        Args:
            rng: Optional Random instance for reproducibility.

        Returns:
            Random GraphArchitecture.
        """
        r = rng or random
        conn = [r.randint(0, GRAPH_MAX_CONN - 1) for _ in range(GRAPH_NUM_NODES)]
        ops = [r.randint(0, GRAPH_NUM_OPS - 1) for _ in range(GRAPH_NUM_NODES)]
        return GraphArchitecture(connectivity=conn, operations=ops)

    def copy(self) -> GraphArchitecture:
        """Create an independent deep copy."""
        return GraphArchitecture(
            connectivity=list(self.connectivity),
            operations=list(self.operations),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "connectivity": list(self.connectivity),
            "operations": list(self.operations),
            "ops_names": [GRAPH_OPERATIONS[o] for o in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphArchitecture:
        """Deserialize from dictionary."""
        return cls(
            connectivity=list(data["connectivity"]),
            operations=list(data["operations"]),
        )

    def to_arch(self) -> Any:
        """Convert to nas_bench_graph.Arch for benchmark lookup.

        Returns:
            nas_bench_graph.Arch object.
        """
        from nas_bench_graph import Arch

        op_names = [GRAPH_OPERATIONS[o] for o in self.operations]
        return Arch(list(self.connectivity), op_names)

    def valid_hash(self) -> int:
        """Compute the valid hash for benchmark lookup.

        Returns:
            Hash integer usable as bench[hash] key.
        """
        return self.to_arch().valid_hash()

    def to_index(self) -> int:
        """Convert to a unique integer index.

        Treats the 8-value encoding as a mixed-radix number:
        connectivity (base 4) + operations (base 9).
        """
        idx = 0
        base = 1
        for c in self.connectivity:
            idx += c * base
            base *= GRAPH_MAX_CONN
        for o in self.operations:
            idx += o * base
            base *= GRAPH_NUM_OPS
        return idx

    @staticmethod
    def from_index(idx: int) -> GraphArchitecture:
        """Create from a unique integer index."""
        conn = []
        for _ in range(GRAPH_NUM_NODES):
            conn.append(idx % GRAPH_MAX_CONN)
            idx //= GRAPH_MAX_CONN
        ops = []
        for _ in range(GRAPH_NUM_NODES):
            ops.append(idx % GRAPH_NUM_OPS)
            idx //= GRAPH_NUM_OPS
        return GraphArchitecture(connectivity=conn, operations=ops)

    def hamming_distance(self, other: GraphArchitecture) -> int:
        """Compute Hamming distance to another architecture.

        Counts positions where connectivity or operations differ.
        """
        dist = sum(a != b for a, b in zip(self.connectivity, other.connectivity))
        dist += sum(a != b for a, b in zip(self.operations, other.operations))
        return dist

    def get_op_name(self, node_idx: int) -> str:
        """Get operation name for a specific computing node."""
        return GRAPH_OPERATIONS[self.operations[node_idx]]

    def num_gnn_ops(self) -> int:
        """Count nodes with GNN operations (gcn, gat, sage, gin, cheb, arma, k-gnn)."""
        return sum(1 for o in self.operations if o <= 6)

    def num_identity(self) -> int:
        """Count nodes with identity (skip) operation."""
        return sum(1 for o in self.operations if o == 7)

    def num_fc(self) -> int:
        """Count nodes with fully connected operation."""
        return sum(1 for o in self.operations if o == 8)

    def __hash__(self) -> int:
        return hash((tuple(self.connectivity), tuple(self.operations)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphArchitecture):
            return NotImplemented
        return (self.connectivity == other.connectivity and
                self.operations == other.operations)

    def __repr__(self) -> str:
        ops = [GRAPH_OPERATIONS[o][:4] for o in self.operations]
        return f"GraphArchitecture(conn={self.connectivity}, ops={ops})"
