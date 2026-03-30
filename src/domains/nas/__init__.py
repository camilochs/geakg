"""NAS Domain: Neural Architecture Search as a GEAKG case study.

This domain treats neural architecture design as a search problem where:
- Solutions are neural architecture DAGs (layers, connections, hyperparameters)
- Operators modify architectures (add layers, change activations, etc.)
- Fitness is validation accuracy (negated for minimization)

Three representations:
- NeuralArchitecture: Layer-based (original, for custom search spaces)
- CellArchitecture: Cell-based (NAS-Bench-201, 5^6 = 15,625 architectures)
- GraphArchitecture: GNN cell (NAS-Bench-Graph, 26,206 unique architectures)
"""

from src.domains.nas.architecture import NeuralArchitecture, ArchitectureLayer
from src.domains.nas.cell_architecture import CellArchitecture, OPERATIONS
from src.domains.nas.graph_architecture import GraphArchitecture, GRAPH_OPERATIONS
from src.domains.nas.context import NASContext, NASBenchContext
from src.domains.nas.graph_context import NASBenchGraphContext
from src.domains.nas.config import NASDomainConfig, NASBenchConfig
from src.domains.nas.search_space import NASSearchSpace
from src.domains.nas.nasbench_evaluator import NASBench201Evaluator
from src.domains.nas.graph_evaluator import NASBenchGraphEvaluator

__all__ = [
    "NeuralArchitecture",
    "ArchitectureLayer",
    "CellArchitecture",
    "OPERATIONS",
    "GraphArchitecture",
    "GRAPH_OPERATIONS",
    "NASContext",
    "NASBenchContext",
    "NASBenchGraphContext",
    "NASDomainConfig",
    "NASBenchConfig",
    "NASSearchSpace",
    "NASBench201Evaluator",
    "NASBenchGraphEvaluator",
]
