"""Family and Domain Registry.

Centralized registration of optimization families and their associated
domains, contexts, and operators.

This module enables:
- Discovery of available families and domains
- Factory creation of contexts by name
- Access to family-specific base operators
- Transfer learning configuration between domains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from src.geakg.contexts.base import OptimizationContext, OptimizationFamily


@dataclass
class DomainInfo:
    """Information about a registered domain."""

    name: str
    """Domain identifier (e.g., 'tsp', 'knapsack')."""

    family: OptimizationFamily
    """Parent family."""

    context_class: Type[OptimizationContext]
    """Context class to instantiate."""

    description: str = ""
    """Human-readable description."""

    instances_dir: str = ""
    """Default directory for instance files."""

    has_optimal: bool = False
    """Whether optimal solutions are typically known."""


@dataclass
class FamilyInfo:
    """Information about a registered family."""

    family: OptimizationFamily
    """Family identifier."""

    description: str
    """Human-readable description."""

    solution_type: str
    """Type annotation for solutions (e.g., 'list[int]')."""

    operators: dict[str, Callable] = field(default_factory=dict)
    """Base operators for this family."""

    domains: dict[str, DomainInfo] = field(default_factory=dict)
    """Registered domains in this family."""

    context_methods: list[str] = field(default_factory=list)
    """Family-specific context methods available to operators."""


class FamilyRegistry:
    """Registry for optimization families and domains.

    Singleton pattern - use FAMILY_REGISTRY instance.
    """

    def __init__(self):
        self._families: dict[OptimizationFamily, FamilyInfo] = {}
        self._domains: dict[str, DomainInfo] = {}

    def register_family(
        self,
        family: OptimizationFamily,
        description: str,
        solution_type: str,
        operators: dict[str, Callable] | None = None,
        context_methods: list[str] | None = None,
    ) -> None:
        """Register an optimization family.

        Args:
            family: Family identifier
            description: Human-readable description
            solution_type: Type annotation for solutions
            operators: Base operators for this family
            context_methods: Family-specific context methods
        """
        self._families[family] = FamilyInfo(
            family=family,
            description=description,
            solution_type=solution_type,
            operators=operators or {},
            context_methods=context_methods or [],
        )

    def register_domain(
        self,
        name: str,
        family: OptimizationFamily,
        context_class: Type[OptimizationContext],
        description: str = "",
        instances_dir: str = "",
        has_optimal: bool = False,
    ) -> None:
        """Register a domain within a family.

        Args:
            name: Domain identifier (e.g., 'tsp')
            family: Parent family
            context_class: Context class to instantiate
            description: Human-readable description
            instances_dir: Default directory for instances
            has_optimal: Whether optimal solutions are known
        """
        info = DomainInfo(
            name=name,
            family=family,
            context_class=context_class,
            description=description,
            instances_dir=instances_dir,
            has_optimal=has_optimal,
        )
        self._domains[name] = info

        # Also register in family
        if family in self._families:
            self._families[family].domains[name] = info

    def get_family(self, family: OptimizationFamily) -> FamilyInfo | None:
        """Get family information."""
        return self._families.get(family)

    def get_domain(self, name: str) -> DomainInfo | None:
        """Get domain information."""
        return self._domains.get(name)

    def get_families(self) -> list[OptimizationFamily]:
        """Get all registered families."""
        return list(self._families.keys())

    def get_domains(self, family: OptimizationFamily | None = None) -> list[str]:
        """Get all registered domains, optionally filtered by family."""
        if family is None:
            return list(self._domains.keys())
        return [
            name for name, info in self._domains.items() if info.family == family
        ]

    def get_operators(self, family: OptimizationFamily) -> dict[str, Callable]:
        """Get base operators for a family."""
        info = self._families.get(family)
        return info.operators if info else {}

    def create_context(self, domain: str, **kwargs: Any) -> OptimizationContext:
        """Create a context instance for a domain.

        Args:
            domain: Domain identifier
            **kwargs: Arguments for context constructor

        Returns:
            Context instance

        Raises:
            ValueError: If domain is not registered
        """
        info = self._domains.get(domain)
        if info is None:
            raise ValueError(f"Unknown domain: {domain}")
        return info.context_class(**kwargs)

    def get_family_for_domain(self, domain: str) -> OptimizationFamily | None:
        """Get the family for a domain."""
        info = self._domains.get(domain)
        return info.family if info else None

    def can_transfer_operators(self, source: str, target: str) -> bool:
        """Check if operators can transfer between domains.

        Operators only transfer within the same family.

        Args:
            source: Source domain
            target: Target domain

        Returns:
            True if operator transfer is possible
        """
        source_info = self._domains.get(source)
        target_info = self._domains.get(target)

        if source_info is None or target_info is None:
            return False

        return source_info.family == target_info.family

    def can_transfer_topology(self, source: str, target: str) -> bool:
        """Check if topology/rules can transfer between domains.

        Topology transfers across all families.

        Args:
            source: Source domain
            target: Target domain

        Returns:
            True if topology transfer is possible
        """
        return source in self._domains and target in self._domains

    def get_context_methods(self, family: OptimizationFamily) -> list[str]:
        """Get family-specific context methods for LLM prompts."""
        info = self._families.get(family)
        return info.context_methods if info else []


# =============================================================================
# Global Registry Instance
# =============================================================================

FAMILY_REGISTRY = FamilyRegistry()


def _initialize_registry() -> None:
    """Initialize the registry with all families and their base operators."""

    # Import operators
    from src.geakg.operators.base.binary import BASE_OPERATORS_BINARY
    from src.geakg.operators.base.continuous import BASE_OPERATORS_CONTINUOUS
    from src.geakg.operators.base.partition import BASE_OPERATORS_PARTITION

    # Register Permutation Family
    FAMILY_REGISTRY.register_family(
        family=OptimizationFamily.PERMUTATION,
        description="Solutions are orderings of elements [0, 1, ..., n-1]. "
        "Operators: swap, insert, reverse (2-opt).",
        solution_type="list[int]",
        operators={},  # Using existing generic_operators/permutation.py
        context_methods=[
            "ctx.swap(solution, i, j) -> list[int]",
            "ctx.insert(solution, i, j) -> list[int]",
            "ctx.reverse(solution, i, j) -> list[int]",
            "ctx.delta_swap(solution, i, j) -> float",
            "ctx.delta_reverse(solution, i, j) -> float",
        ],
    )

    # Register Binary Family
    FAMILY_REGISTRY.register_family(
        family=OptimizationFamily.BINARY,
        description="Solutions are bit vectors [0, 1, 0, 1, ...]. "
        "Operators: flip, swap 0<->1.",
        solution_type="list[int]",
        operators=BASE_OPERATORS_BINARY,
        context_methods=[
            "ctx.flip(solution, i) -> list[int]",
            "ctx.flip_multiple(solution, indices) -> list[int]",
            "ctx.delta_flip(solution, i) -> float",
            "ctx.get_selected_indices(solution) -> list[int]",
            "ctx.get_unselected_indices(solution) -> list[int]",
            "ctx.repair_greedy(solution) -> list[int]",
        ],
    )

    # Register Continuous Family
    FAMILY_REGISTRY.register_family(
        family=OptimizationFamily.CONTINUOUS,
        description="Solutions are real-valued vectors [x1, x2, ..., xn]. "
        "Operators: perturb, gradient step.",
        solution_type="list[float]",
        operators=BASE_OPERATORS_CONTINUOUS,
        context_methods=[
            "ctx.perturb(solution, sigma) -> list[float]",
            "ctx.gradient(solution) -> list[float]",
            "ctx.gradient_step(solution, gradient, step_size) -> list[float]",
            "ctx.clip(solution) -> list[float]",
            "ctx.bounds -> list[tuple[float, float]]",
            "ctx.crossover_blend(p1, p2, alpha) -> list[float]",
        ],
    )

    # Register Partition Family
    FAMILY_REGISTRY.register_family(
        family=OptimizationFamily.PARTITION,
        description="Solutions assign items to groups/bins. "
        "Operators: move, swap_items, merge_groups.",
        solution_type="list[int]",
        operators=BASE_OPERATORS_PARTITION,
        context_methods=[
            "ctx.move(solution, item, to_group) -> list[int]",
            "ctx.swap_items(solution, i, j) -> list[int]",
            "ctx.merge_groups(solution, g1, g2) -> list[int]",
            "ctx.group_load(solution, group) -> float",
            "ctx.balance_metric(solution) -> float",
            "ctx.get_groups(solution) -> dict[int, list[int]]",
        ],
    )


# Initialize on import
_initialize_registry()


__all__ = [
    "FamilyRegistry",
    "FamilyInfo",
    "DomainInfo",
    "FAMILY_REGISTRY",
]
