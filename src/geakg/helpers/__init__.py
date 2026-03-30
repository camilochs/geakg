"""Domain-specific helper functions for synthesized operators.

These helpers provide a transferable interface:
- LLM generates code using helpers with (solution, data, ...) signature
- Helpers are implemented differently per domain
- Same operator code works across domains if it uses helpers

For transfer learning, the key insight is:
- position_cost(tour, dm, i) in TSP = position_cost(schedule, times, i) in JSSP
- The semantic meaning is the same, implementation differs
"""

from src.geakg.helpers.tsp import TSP_HELPERS, get_tsp_helper_namespace

__all__ = ["TSP_HELPERS", "get_tsp_helper_namespace"]
