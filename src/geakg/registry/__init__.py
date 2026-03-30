"""Registry for families, domains, and operators.

Provides centralized registration and discovery of:
- Optimization families (permutation, binary, continuous, partition)
- Domain contexts (TSP, Knapsack, etc.)
- Base operators for each family
"""

from src.geakg.registry.families import FamilyRegistry, FAMILY_REGISTRY

__all__ = ["FamilyRegistry", "FAMILY_REGISTRY"]
