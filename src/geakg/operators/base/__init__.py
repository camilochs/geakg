"""Base operators for each optimization family.

Each family has 11 base operators covering the 3 categories:
- Construction (4 operators)
- Local Search (4 operators)
- Perturbation (3 operators)

These serve as initial operators for the L0 pool and provide
starting points for LLM-guided synthesis.
"""

from src.geakg.operators.base.binary import BASE_OPERATORS_BINARY
from src.geakg.operators.base.continuous import BASE_OPERATORS_CONTINUOUS
from src.geakg.operators.base.partition import BASE_OPERATORS_PARTITION

__all__ = [
    "BASE_OPERATORS_BINARY",
    "BASE_OPERATORS_CONTINUOUS",
    "BASE_OPERATORS_PARTITION",
]
