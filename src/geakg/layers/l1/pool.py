"""Operator pool data structures for L1 operator generation.

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/  <-- THIS FILE
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Operator:
    """A single operator with code and metadata."""

    name: str
    code: str
    role: str
    design_choices: dict[str, str] = field(default_factory=dict)
    interaction_effects: str = ""
    fitness_scores: list[float] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def avg_fitness(self) -> float:
        """Average fitness across all evaluations."""
        if not self.fitness_scores:
            return float("inf")
        return sum(self.fitness_scores) / len(self.fitness_scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "role": self.role,
            "design_choices": self.design_choices,
            "interaction_effects": self.interaction_effects,
            "fitness_scores": self.fitness_scores,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Operator:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            code=data["code"],
            role=data["role"],
            design_choices=data.get("design_choices", {}),
            interaction_effects=data.get("interaction_effects", ""),
            fitness_scores=data.get("fitness_scores", []),
        )


@dataclass
class OperatorPool:
    """Pool of operators organized by role.

    The pool is generated offline by L1 and loaded at runtime.
    Each role has multiple operators that can be selected by ACO.
    """

    operators_by_role: dict[str, list[Operator]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_operator(self, operator: Operator) -> None:
        """Add an operator to its role's pool."""
        if operator.role not in self.operators_by_role:
            self.operators_by_role[operator.role] = []
        self.operators_by_role[operator.role].append(operator)

    def remove_operator(self, name: str) -> bool:
        """Remove an operator by name.

        Args:
            name: Operator name to remove.

        Returns:
            True if operator was found and removed, False otherwise.
        """
        for role, operators in self.operators_by_role.items():
            for i, op in enumerate(operators):
                if op.name == name:
                    operators.pop(i)
                    return True
        return False

    def get_operators_for_role(self, role: str) -> list[Operator]:
        """Get all operators for a specific role."""
        return self.operators_by_role.get(role, [])

    def get_best_for_role(self, role: str) -> Operator | None:
        """Get the best operator for a role (by average fitness)."""
        operators = self.get_operators_for_role(role)
        if not operators:
            return None
        return min(operators, key=lambda op: op.avg_fitness)

    def get_operator_by_name(self, name: str) -> Operator | None:
        """Find operator by name across all roles.

        Args:
            name: Operator name to search for.

        Returns:
            Operator if found, None otherwise.
        """
        for operators in self.operators_by_role.values():
            for op in operators:
                if op.name == name:
                    return op
        return None

    @property
    def roles(self) -> list[str]:
        """List of all roles in the pool."""
        return list(self.operators_by_role.keys())

    @property
    def total_operators(self) -> int:
        """Total number of operators in the pool."""
        return sum(len(ops) for ops in self.operators_by_role.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize pool to dictionary."""
        return {
            "metadata": self.metadata,
            "operators_by_role": {
                role: [op.to_dict() for op in operators]
                for role, operators in self.operators_by_role.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorPool:
        """Deserialize pool from dictionary."""
        pool = cls(metadata=data.get("metadata", {}))
        for role, operators_data in data.get("operators_by_role", {}).items():
            for op_data in operators_data:
                pool.add_operator(Operator.from_dict(op_data))
        return pool

    def save(self, path: Path | str) -> None:
        """Save pool to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> OperatorPool:
        """Load pool from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_bindings(self) -> dict[str, str]:
        """Convert to role->code bindings for metagraph.

        Returns the best operator's code for each role.
        """
        bindings = {}
        for role in self.roles:
            best = self.get_best_for_role(role)
            if best:
                bindings[role] = best.code
        return bindings

    def __repr__(self) -> str:
        return (
            f"OperatorPool(roles={len(self.roles)}, "
            f"total_operators={self.total_operators})"
        )
