"""NAS-Bench-Graph Context: optimization context for GNN architecture search.

Wraps NASBenchGraphEvaluator to provide the standard OptimizationContext
interface for ACO traversal and operator application.

Metric: accuracy (higher is better) -> returns -accuracy for minimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.geakg.contexts.base import OptimizationContext, OptimizationFamily

if TYPE_CHECKING:
    from src.domains.nas.graph_architecture import GraphArchitecture
    from src.domains.nas.graph_evaluator import NASBenchGraphEvaluator


class NASBenchGraphContext(OptimizationContext["GraphArchitecture"]):
    """Context for NAS-Bench-Graph GNN architecture search.

    Uses GraphArchitecture (4 connectivity + 4 operations) as solution type.
    Evaluation via NAS-Bench-Graph tabular lookup (O(1) per evaluation).

    Fitness = -accuracy (minimization convention, same as NAS-Bench-201).
    """

    def __init__(
        self,
        evaluator: "NASBenchGraphEvaluator",
    ) -> None:
        self._evaluator = evaluator

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.ARCHITECTURE

    @property
    def domain(self) -> str:
        return "nas_bench_graph"

    @property
    def dimension(self) -> int:
        """Dimension = 8 (4 connectivity + 4 operations)."""
        return 8

    def evaluate(self, solution: "GraphArchitecture") -> float:
        """Evaluate architecture via NAS-Bench-Graph lookup.

        Returns -accuracy (minimization convention).
        """
        accuracy = self._evaluator.evaluate(solution)
        return -accuracy

    def valid(self, solution: "GraphArchitecture") -> bool:
        """Check if graph architecture is valid."""
        from src.domains.nas.graph_architecture import (
            GRAPH_NUM_NODES,
            GRAPH_NUM_OPS,
            GRAPH_MAX_CONN,
        )

        if len(solution.connectivity) != GRAPH_NUM_NODES:
            return False
        if len(solution.operations) != GRAPH_NUM_NODES:
            return False
        if not all(0 <= c < GRAPH_MAX_CONN for c in solution.connectivity):
            return False
        return all(0 <= o < GRAPH_NUM_OPS for o in solution.operations)

    def random_solution(self) -> "GraphArchitecture":
        """Generate a random valid graph architecture."""
        from src.domains.nas.graph_architecture import GraphArchitecture

        return GraphArchitecture.random()

    def copy(self, solution: "GraphArchitecture") -> "GraphArchitecture":
        """Deep copy of graph architecture."""
        return solution.copy()

    @property
    def instance_data(self) -> dict:
        """Instance data for NAS-Bench-Graph."""
        return {
            "dimension": self.dimension,
            "dataset": self._evaluator.dataset,
            "search_space": "NAS-Bench-Graph",
            "metric": "accuracy",
            "direction": "maximize",
        }
