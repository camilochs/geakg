"""NAS-Bench-201 Cell Architecture representation.

NAS-Bench-201 uses a cell-based search space:
- 4 nodes, 6 directed edges
- Each edge selects from 5 operations: {none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3}
- Total: 5^6 = 15,625 unique architectures

Reference: Dong & Yang, "NAS-Bench-201: Extending the Scope of Reproducible
Neural Architecture Search", ICLR 2020.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# NAS-Bench-201 operation vocabulary (order matters for encoding)
OPERATIONS = [
    "none",           # 0: no connection
    "skip_connect",   # 1: identity / skip connection
    "nor_conv_1x1",   # 2: 1x1 convolution (ReLU-Conv-BN)
    "nor_conv_3x3",   # 3: 3x3 convolution (ReLU-Conv-BN)
    "avg_pool_3x3",   # 4: 3x3 average pooling
]

NUM_EDGES = 6
NUM_OPS = len(OPERATIONS)


@dataclass
class CellArchitecture:
    """A cell architecture in the NAS-Bench-201 search space.

    Represents a DAG with 4 nodes and 6 directed edges.
    Edge (i,j) connects node i to node j where i < j:
        edges[0] = (0→1)
        edges[1] = (0→2)
        edges[2] = (1→2)
        edges[3] = (0→3)
        edges[4] = (1→3)
        edges[5] = (2→3)

    Each edge value is an integer in [0, 4] indexing into OPERATIONS.
    """

    edges: list[int] = field(default_factory=lambda: [0] * NUM_EDGES)

    def __post_init__(self) -> None:
        if len(self.edges) != NUM_EDGES:
            raise ValueError(f"Expected {NUM_EDGES} edges, got {len(self.edges)}")
        for i, e in enumerate(self.edges):
            if not (0 <= e < NUM_OPS):
                raise ValueError(
                    f"Edge {i} has value {e}, must be in [0, {NUM_OPS - 1}]"
                )

    def to_nasbench_string(self) -> str:
        """Convert to the NAS-Bench-201 architecture string format.

        Format: |op~node|+|op~node|op~node|+|op~node|op~node|op~node|

        Returns:
            Architecture string compatible with NAS-Bench-201 API.
        """
        ops = [OPERATIONS[e] for e in self.edges]
        return (
            f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
        )

    @staticmethod
    def from_nasbench_string(arch_str: str) -> CellArchitecture:
        """Parse a NAS-Bench-201 architecture string.

        Args:
            arch_str: Architecture string in NAS-Bench-201 format.

        Returns:
            CellArchitecture with decoded edges.
        """
        # Extract operations between | delimiters, ignoring + separators
        parts = arch_str.split("|")
        ops_found: list[str] = []
        for part in parts:
            part = part.strip().strip("+").strip()
            if "~" in part:
                op_name = part.split("~")[0]
                ops_found.append(op_name)
        edges = [OPERATIONS.index(op) for op in ops_found]
        return CellArchitecture(edges=edges)

    @staticmethod
    def random(rng: random.Random | None = None) -> CellArchitecture:
        """Generate a random cell architecture.

        Args:
            rng: Optional Random instance for reproducibility.

        Returns:
            Random CellArchitecture.
        """
        r = rng or random
        edges = [r.randint(0, NUM_OPS - 1) for _ in range(NUM_EDGES)]
        return CellArchitecture(edges=edges)

    def copy(self) -> CellArchitecture:
        """Create an independent deep copy."""
        return CellArchitecture(edges=list(self.edges))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "edges": list(self.edges),
            "arch_string": self.to_nasbench_string(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CellArchitecture:
        """Deserialize from dictionary."""
        return cls(edges=list(data["edges"]))

    def to_index(self) -> int:
        """Convert to a unique integer index in [0, 15624].

        Treats edges as a base-5 number.
        """
        idx = 0
        for i, e in enumerate(self.edges):
            idx += e * (NUM_OPS ** i)
        return idx

    @staticmethod
    def from_index(idx: int) -> CellArchitecture:
        """Create from a unique integer index.

        Args:
            idx: Integer in [0, 15624].
        """
        edges = []
        for _ in range(NUM_EDGES):
            edges.append(idx % NUM_OPS)
            idx //= NUM_OPS
        return CellArchitecture(edges=edges)

    def get_op_name(self, edge_idx: int) -> str:
        """Get operation name for a specific edge."""
        return OPERATIONS[self.edges[edge_idx]]

    def num_non_none_edges(self) -> int:
        """Count edges that are not 'none'."""
        return sum(1 for e in self.edges if e != 0)

    def __hash__(self) -> int:
        return hash(tuple(self.edges))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellArchitecture):
            return NotImplemented
        return self.edges == other.edges

    def __repr__(self) -> str:
        ops = [OPERATIONS[e][:4] for e in self.edges]
        return f"CellArchitecture({ops})"
