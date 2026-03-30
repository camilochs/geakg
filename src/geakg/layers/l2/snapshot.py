"""GEAKG Snapshot: Serializable state of the entire knowledge graph.

A GEAKGSnapshot contains all three layers:
- L0: MetaGraph topology (roles, transitions, initial weights)
- L1: Operator pool (executable code for each role)
- L2: Learned knowledge (pheromones, symbolic rules)

This snapshot is used for:
- Transfer learning: Load TSP-trained knowledge for VRP
- Persistence: Save/load training progress
- Online execution: Symbolic executor loads snapshot without LLM

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/  <-- THIS FILE
    Online: Symbolic Executor - src/geakg/online/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.geakg.layers.l0.metagraph import MetaGraph
    from src.geakg.layers.l1.pool import OperatorPool
    from src.geakg.layers.l2.pheromones import PheromoneMatrix


@dataclass
class GEAKGSnapshot:
    """Complete snapshot of a trained GEAKG.

    Contains all information needed to run the symbolic executor
    without any LLM calls. Used for transfer learning and deployment.

    Attributes:
        name: Snapshot name (e.g., "tsp_50k_trained")
        domain: Source domain (e.g., "tsp")
        version: Snapshot format version
        l0_topology: MetaGraph topology as dict (roles, edges, conditions)
        l1_pool_ref: Reference to operator pool file
        l1_operators_by_role: Inline operators (optional, for self-contained snapshots)
        l2_pheromones: Learned pheromone values
        l2_symbolic_rules: Extracted symbolic rules
        metadata: Training metadata (iterations, best gap, etc.)
    """

    name: str
    domain: str
    version: str = "2.0"  # GEAKG format version

    # L0: Topology
    l0_topology: dict[str, Any] = field(default_factory=dict)

    # L1: Operators
    l1_pool_ref: str | None = None  # Path to pool file
    l1_operators_by_role: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # L2: Learned knowledge
    l2_pheromones: dict[str, Any] = field(default_factory=dict)
    l2_symbolic_rules: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "l0_topology": self.l0_topology,
            "l1_pool_ref": self.l1_pool_ref,
            "l1_operators_by_role": self.l1_operators_by_role,
            "l2_pheromones": self.l2_pheromones,
            "l2_symbolic_rules": self.l2_symbolic_rules,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GEAKGSnapshot:
        """Deserialize snapshot from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            domain=data.get("domain", "unknown"),
            version=data.get("version", "1.0"),
            l0_topology=data.get("l0_topology", data.get("metagraph", {})),
            l1_pool_ref=data.get("l1_pool_ref"),
            l1_operators_by_role=data.get(
                "l1_operators_by_role",
                data.get("operators_by_role", {})
            ),
            l2_pheromones=data.get("l2_pheromones", data.get("pheromones", {})),
            l2_symbolic_rules=data.get("l2_symbolic_rules", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save snapshot to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved GEAKG snapshot to {path}")

    @classmethod
    def load(cls, path: Path | str) -> GEAKGSnapshot:
        """Load snapshot from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded GEAKGSnapshot
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        snapshot = cls.from_dict(data)
        logger.info(f"Loaded GEAKG snapshot '{snapshot.name}' from {path}")
        return snapshot

    def get_metagraph(self) -> "MetaGraph":
        """Reconstruct MetaGraph from L0 topology.

        Returns:
            MetaGraph object
        """
        from src.geakg.layers.l0.metagraph import MetaGraph
        return MetaGraph.from_dict(self.l0_topology)

    def get_operator_pool(self) -> "OperatorPool":
        """Get operator pool (load from ref or use inline).

        Returns:
            OperatorPool object
        """
        from src.geakg.layers.l1.pool import OperatorPool

        if self.l1_operators_by_role:
            return OperatorPool.from_dict({
                "operators_by_role": self.l1_operators_by_role,
                "metadata": {"source": "snapshot_inline"},
            })

        if self.l1_pool_ref:
            return OperatorPool.load(self.l1_pool_ref)

        raise ValueError("No operator pool available in snapshot")

    def get_pheromone_matrix(self) -> "PheromoneMatrix":
        """Get pheromone matrix from L2 learned knowledge.

        Returns:
            PheromoneMatrix object
        """
        from src.geakg.layers.l2.pheromones import PheromoneMatrix
        return PheromoneMatrix.from_dict(self.l2_pheromones)

    @classmethod
    def from_training(
        cls,
        name: str,
        domain: str,
        metagraph: "MetaGraph",
        pool: "OperatorPool",
        pheromones: "PheromoneMatrix",
        symbolic_rules: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GEAKGSnapshot:
        """Create snapshot from training results.

        Args:
            name: Snapshot name
            domain: Source domain
            metagraph: Trained MetaGraph
            pool: Operator pool
            pheromones: Learned pheromones
            symbolic_rules: Extracted rules (optional)
            metadata: Training metadata (optional)

        Returns:
            New GEAKGSnapshot
        """
        # Inline operators for self-contained snapshot
        operators_by_role = {}
        for role in pool.roles:
            operators_by_role[role] = [
                op.to_dict() for op in pool.get_operators_for_role(role)
            ]

        return cls(
            name=name,
            domain=domain,
            l0_topology=metagraph.to_dict(),
            l1_operators_by_role=operators_by_role,
            l2_pheromones=pheromones.to_dict(),
            l2_symbolic_rules=symbolic_rules or [],
            metadata=metadata or {},
        )


# Backward compatibility alias
AKGSnapshot = GEAKGSnapshot
