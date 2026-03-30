"""Representation types and generic operators for NS-SE.

This module defines the representation-based operator system that enables
zero-shot transfer to any domain with a known representation type.

Instead of 118 domain-specific operators, we use 21 generic operators
for permutation-based problems. Level 4 synthesizes domain-specific
operators when needed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from akg.roles import AbstractRole


class RepresentationType(Enum):
    """Types of solution representations.

    Each representation type has a set of generic operators that work
    on any problem with that representation.
    """

    PERMUTATION = "permutation"  # TSP, JSSP, QAP, PFSP
    BINARY_VECTOR = "binary_vector"  # Knapsack, Set Cover
    INTEGER_SEQUENCE = "integer_sequence"  # Scheduling with repetition
    PARTITION = "partition"  # BPP, Graph Coloring
    CONTINUOUS = "continuous"  # Numerical optimization
    ARCHITECTURE_DAG = "architecture_dag"  # NAS: neural architecture as DAG


@dataclass
class GenericOperator:
    """A generic operator that works on a specific representation type.

    Generic operators are representation-aware but domain-agnostic.
    They can be applied to any problem with matching representation.

    Attributes:
        operator_id: Unique identifier for the operator.
        function: The operator implementation function.
        role: The abstract role this operator fulfills.
        weight: Selection weight for ACO (higher = more likely).
        description: Human-readable description.
        representation: The representation type this operator works on.
    """

    operator_id: str
    function: Callable[..., Any]
    role: str  # Role name as string to avoid circular import
    weight: float = 1.0
    description: str = ""
    representation: RepresentationType = RepresentationType.PERMUTATION


@dataclass
class DomainSpec:
    """Specification for a problem domain.

    A domain is defined by its representation type and fitness function.
    This enables zero-config domain registration - just provide a fitness
    function and the system automatically uses appropriate generic operators.

    Attributes:
        name: Domain identifier (e.g., "tsp", "qap").
        representation: Type of solution representation.
        fitness_fn: Function to evaluate solution quality (lower is better).
        solution_size_fn: Function to get solution size from instance.
        validator: Optional function to validate solutions.
        use_domain_specific: Whether to use predefined domain operators (warm start).
    """

    name: str
    representation: RepresentationType
    fitness_fn: Callable[[Any, Any], float]
    solution_size_fn: Callable[[Any], int]
    validator: Callable[[Any], bool] | None = None
    use_domain_specific: bool = False  # Disabled by default


@dataclass
class RepresentationOperators:
    """Collection of generic operators for a representation type.

    Organizes operators by their abstract role, making it easy to
    create bindings for any domain with matching representation.
    """

    representation: RepresentationType
    operators_by_role: dict[str, list[GenericOperator]] = field(default_factory=dict)

    def add_operator(self, operator: GenericOperator) -> None:
        """Add an operator to the collection."""
        if operator.role not in self.operators_by_role:
            self.operators_by_role[operator.role] = []
        self.operators_by_role[operator.role].append(operator)

    def get_operators(self, role: str) -> list[GenericOperator]:
        """Get all operators for a given role."""
        return self.operators_by_role.get(role, [])

    def get_all_operators(self) -> list[GenericOperator]:
        """Get all operators across all roles."""
        result = []
        for ops in self.operators_by_role.values():
            result.extend(ops)
        return result

    def count_operators(self) -> int:
        """Count total number of operators."""
        return sum(len(ops) for ops in self.operators_by_role.values())

    def count_roles(self) -> int:
        """Count number of roles with at least one operator."""
        return len(self.operators_by_role)
