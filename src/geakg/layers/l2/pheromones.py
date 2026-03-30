"""Pheromone structures for L2 learned knowledge.

Pheromones represent empirical knowledge learned through ACO training.
They refine the initial weights from L0 (LLM heuristics) based on
successful paths discovered during training.

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/  <-- THIS FILE
    Online: Symbolic Executor - src/geakg/online/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PheromoneEntry:
    """A single pheromone entry for a role-operator combination.

    Attributes:
        role: The role ID (e.g., "ls_intensify_small")
        operator: The operator name
        value: Pheromone value (higher = better historically)
        uses: Number of times this operator was used
        successes: Number of times it led to improvement
    """
    role: str
    operator: str
    value: float
    uses: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate of this operator."""
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role,
            "operator": self.operator,
            "value": self.value,
            "uses": self.uses,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PheromoneEntry:
        """Deserialize from dictionary."""
        return cls(
            role=data["role"],
            operator=data["operator"],
            value=data["value"],
            uses=data.get("uses", 0),
            successes=data.get("successes", 0),
        )


@dataclass
class PheromoneMatrix:
    """Matrix of pheromone values for role-operator combinations.

    The matrix stores learned preferences for:
    - Which operator to use for each role
    - Which role to transition to next (edge pheromones)

    These values are learned during ACO training (L2) and used
    by the symbolic executor (online phase) to guide search.
    """

    # Operator pheromones: role -> operator -> value
    operator_pheromones: dict[str, dict[str, float]] = field(default_factory=dict)

    # Edge pheromones: (source_role, target_role) -> value
    edge_pheromones: dict[tuple[str, str], float] = field(default_factory=dict)

    # Metadata about training
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_operator_pheromone(self, role: str, operator: str) -> float:
        """Get pheromone value for a role-operator combination.

        Args:
            role: Role ID
            operator: Operator name

        Returns:
            Pheromone value (default 1.0 if not set)
        """
        if role not in self.operator_pheromones:
            return 1.0
        return self.operator_pheromones[role].get(operator, 1.0)

    def set_operator_pheromone(self, role: str, operator: str, value: float) -> None:
        """Set pheromone value for a role-operator combination.

        Args:
            role: Role ID
            operator: Operator name
            value: New pheromone value
        """
        if role not in self.operator_pheromones:
            self.operator_pheromones[role] = {}
        self.operator_pheromones[role][operator] = value

    def get_edge_pheromone(self, source: str, target: str) -> float:
        """Get pheromone value for a role transition.

        Args:
            source: Source role ID
            target: Target role ID

        Returns:
            Pheromone value (default 1.0 if not set)
        """
        return self.edge_pheromones.get((source, target), 1.0)

    def set_edge_pheromone(self, source: str, target: str, value: float) -> None:
        """Set pheromone value for a role transition.

        Args:
            source: Source role ID
            target: Target role ID
            value: New pheromone value
        """
        self.edge_pheromones[(source, target)] = value

    def evaporate(self, rate: float = 0.1) -> None:
        """Apply pheromone evaporation to all values.

        Args:
            rate: Evaporation rate (0.1 = 10% reduction)
        """
        factor = 1.0 - rate

        for role in self.operator_pheromones:
            for operator in self.operator_pheromones[role]:
                self.operator_pheromones[role][operator] *= factor

        for key in self.edge_pheromones:
            self.edge_pheromones[key] *= factor

    def deposit(
        self,
        path: list[tuple[str, str]],
        quality: float,
        deposit_amount: float = 1.0,
    ) -> None:
        """Deposit pheromones along a successful path.

        Args:
            path: List of (role, operator) tuples
            quality: Solution quality (higher = better path)
            deposit_amount: Base amount to deposit
        """
        amount = deposit_amount * quality

        for role, operator in path:
            current = self.get_operator_pheromone(role, operator)
            self.set_operator_pheromone(role, operator, current + amount)

        # Deposit on edges
        for i in range(len(path) - 1):
            source_role = path[i][0]
            target_role = path[i + 1][0]
            current = self.get_edge_pheromone(source_role, target_role)
            self.set_edge_pheromone(source_role, target_role, current + amount)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operator_pheromones": self.operator_pheromones,
            "edge_pheromones": {
                f"{k[0]}|{k[1]}": v
                for k, v in self.edge_pheromones.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PheromoneMatrix:
        """Deserialize from dictionary."""
        edge_pheromones = {}
        for key, value in data.get("edge_pheromones", {}).items():
            parts = key.split("|")
            if len(parts) == 2:
                edge_pheromones[(parts[0], parts[1])] = value

        return cls(
            operator_pheromones=data.get("operator_pheromones", {}),
            edge_pheromones=edge_pheromones,
            metadata=data.get("metadata", {}),
        )

    def get_best_operators(self, top_k: int = 3) -> dict[str, list[str]]:
        """Get top-k operators for each role by pheromone value.

        Args:
            top_k: Number of top operators to return per role

        Returns:
            Dict mapping role -> list of top operator names
        """
        result = {}
        for role, operators in self.operator_pheromones.items():
            sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)
            result[role] = [op for op, _ in sorted_ops[:top_k]]
        return result
