"""NAS Context: Implements the optimization context interface for NAS.

Two context implementations:
- NASContext: Original, works with NeuralArchitecture (layer-based)
- NASBenchContext: NAS-Bench-201, works with CellArchitecture (cell-based)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.geakg.contexts.base import OptimizationContext, OptimizationFamily
from src.domains.nas.architecture import NeuralArchitecture
from src.domains.nas.search_space import NASSearchSpace

if TYPE_CHECKING:
    from src.domains.nas.evaluator import ArchitectureEvaluator
    from src.domains.nas.cell_architecture import CellArchitecture
    from src.domains.nas.nasbench_evaluator import NASBench201Evaluator


class NASContext(OptimizationContext[NeuralArchitecture]):
    """Context for Neural Architecture Search.

    Wraps the architecture evaluator to provide the standard
    context interface: evaluate, valid, random_solution, copy.

    Fitness = negative validation accuracy (lower is better,
    consistent with minimization convention).
    """

    def __init__(
        self,
        search_space: NASSearchSpace,
        evaluator: "ArchitectureEvaluator | None" = None,
        dataset: str = "cifar10",
    ) -> None:
        self._search_space = search_space
        self._evaluator = evaluator
        self._dataset = dataset

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.ARCHITECTURE

    @property
    def domain(self) -> str:
        return "nas"

    @property
    def dimension(self) -> int:
        """Dimension = max layers in search space."""
        return self._search_space.max_layers

    def evaluate(self, solution: NeuralArchitecture) -> float:
        """Evaluate architecture quality.

        Returns negative validation accuracy (lower is better).
        If no evaluator is set, uses a proxy based on architecture
        properties (parameter count, depth, etc.).
        """
        if self._evaluator is not None:
            accuracy = self._evaluator.evaluate(solution)
            return -accuracy  # Negate: lower is better

        # Proxy evaluation based on architecture properties
        return self._proxy_evaluate(solution)

    def _proxy_evaluate(self, arch: NeuralArchitecture) -> float:
        """Simple proxy evaluation without training.

        Estimates quality based on architecture properties:
        - Moderate depth is better than very shallow or very deep
        - Skip connections help
        - Modern activations score better
        - Regularization helps
        """
        score = 0.0

        # Depth bonus (3-6 layers optimal)
        depth = arch.depth()
        if 3 <= depth <= 6:
            score += 0.3
        elif depth < 3:
            score += 0.1
        else:
            score += 0.2

        # Skip connections bonus
        if arch.skip_connections:
            score += 0.1 * min(len(arch.skip_connections), 3)

        # Modern activation bonus
        modern = {"gelu", "silu", "mish"}
        modern_count = sum(1 for l in arch.layers if l.activation in modern)
        score += 0.05 * modern_count

        # Normalization bonus
        has_norm = any(l.normalization != "none" for l in arch.layers)
        if has_norm:
            score += 0.1

        # Dropout bonus (moderate)
        avg_dropout = sum(l.dropout for l in arch.layers) / max(len(arch.layers), 1)
        if 0.1 <= avg_dropout <= 0.3:
            score += 0.1

        # Optimizer bonus
        if arch.optimizer in ("adamw", "adam"):
            score += 0.1

        # Parameter efficiency
        params = arch.total_params()
        if params < 1_000_000:
            score += 0.1  # Efficient
        elif params > 5_000_000:
            score -= 0.1  # Too large

        # Add small random noise for diversity
        import random
        score += random.uniform(-0.02, 0.02)

        return -score  # Negate: lower is better

    def valid(self, solution: NeuralArchitecture) -> bool:
        """Check if architecture is valid."""
        return self._search_space.is_valid(solution)

    def random_solution(self) -> NeuralArchitecture:
        """Generate a random valid architecture."""
        return self._search_space.random_architecture()

    def copy(self, solution: NeuralArchitecture) -> NeuralArchitecture:
        """Deep copy of architecture."""
        return solution.copy()

    @property
    def instance_data(self) -> dict:
        """Instance data for NAS."""
        return {
            "dimension": self.dimension,
            "dataset": self._dataset,
            "max_params": self._search_space.max_params,
            "max_layers": self._search_space.max_layers,
        }


class NASBenchContext(OptimizationContext["CellArchitecture"]):
    """Context for NAS-Bench-201 cell-based architecture search.

    Uses CellArchitecture (6 edges, 5 operations) as solution type.
    Evaluation via NAS-Bench-201 tabular lookup (O(1) per evaluation).

    Fitness = negative test accuracy (lower is better,
    consistent with minimization convention).
    """

    def __init__(
        self,
        evaluator: "NASBench201Evaluator",
        dataset: str = "cifar10",
    ) -> None:
        self._evaluator = evaluator
        self._dataset = dataset

    @property
    def family(self) -> OptimizationFamily:
        return OptimizationFamily.ARCHITECTURE

    @property
    def domain(self) -> str:
        return "nas_bench"

    @property
    def dimension(self) -> int:
        """Dimension = number of edges in cell (6)."""
        return 6

    def evaluate(self, solution: "CellArchitecture") -> float:
        """Evaluate architecture via NAS-Bench-201 lookup.

        Returns negative test accuracy (lower is better).
        """
        accuracy = self._evaluator.evaluate(solution, self._dataset)
        return -accuracy  # Negate: lower is better

    def valid(self, solution: "CellArchitecture") -> bool:
        """Check if cell architecture is valid."""
        from src.domains.nas.cell_architecture import NUM_EDGES, NUM_OPS
        if len(solution.edges) != NUM_EDGES:
            return False
        return all(0 <= e < NUM_OPS for e in solution.edges)

    def random_solution(self) -> "CellArchitecture":
        """Generate a random valid cell architecture."""
        from src.domains.nas.cell_architecture import CellArchitecture
        return CellArchitecture.random()

    def copy(self, solution: "CellArchitecture") -> "CellArchitecture":
        """Deep copy of cell architecture."""
        return solution.copy()

    @property
    def instance_data(self) -> dict:
        """Instance data for NAS-Bench-201."""
        return {
            "dimension": self.dimension,
            "dataset": self._dataset,
            "search_space": "NAS-Bench-201",
            "total_architectures": 15625,
        }
