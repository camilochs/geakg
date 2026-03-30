"""Domain context adapters for domain-agnostic operators.

Each context adapter implements the OptimizationContext protocol,
hiding domain-specific data structures (distance_matrix, demands, etc.)
behind a universal interface.

Architecture:
    OptimizationContext (Universal)
           ↓
    FamilyContext (Permutation, Binary, Continuous, Partition)
           ↓
    DomainContext (TSP, Knapsack, etc.)

This enables operators trained on one domain (e.g., TSP) to be
directly executed on another domain (e.g., VRP) without code modification.
"""

# Base classes
from src.geakg.contexts.base import (
    OptimizationContext,
    OptimizationFamily,
    FamilyContext,
)

# Family contexts
from src.geakg.contexts.families import (
    PermutationContext,
    BinaryContext,
    ContinuousContext,
    PartitionContext,
)

# Domain contexts
from src.geakg.contexts.tsp import TSPContext
from src.geakg.contexts.vrp import VRPContext

__all__ = [
    # Base
    "OptimizationContext",
    "OptimizationFamily",
    "FamilyContext",
    # Families
    "PermutationContext",
    "BinaryContext",
    "ContinuousContext",
    "PartitionContext",
    # Domains
    "TSPContext",
    "VRPContext",
]
