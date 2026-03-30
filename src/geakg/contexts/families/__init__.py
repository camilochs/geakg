"""Family-specific optimization contexts.

Each family defines a solution representation and associated operations:
- Permutation: orderings (TSP, JSSP, VRP, PFSP)
- Binary: bit vectors (Knapsack, Set Cover, Feature Selection)
- Continuous: real vectors (Function Optimization)
- Partition: group assignments (Bin Packing, Clustering)
"""

from src.geakg.contexts.families.permutation import PermutationContext
from src.geakg.contexts.families.binary import BinaryContext
from src.geakg.contexts.families.continuous import ContinuousContext
from src.geakg.contexts.families.partition import PartitionContext

__all__ = [
    "PermutationContext",
    "BinaryContext",
    "ContinuousContext",
    "PartitionContext",
]
