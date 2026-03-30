"""NAS Domain Configuration for GEAKG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.geakg.representations import RepresentationType
from src.domains.nas.search_space import NASSearchSpace
from src.domains.nas.context import NASContext


@dataclass
class NASDomainConfig:
    """Domain configuration for Neural Architecture Search.

    Analogous to DomainConfig for optimization problems,
    but with NAS-specific settings.
    """

    dataset: str = "cifar10"
    representation_type: RepresentationType = RepresentationType.ARCHITECTURE_DAG
    search_space: NASSearchSpace | None = None
    proxy_epochs: int = 20
    proxy_data_fraction: float = 0.25
    use_gpu: bool = True

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = NASSearchSpace()

    def create_context(self, instance_data: dict | None = None) -> NASContext:
        """Create a NAS context for evaluation."""
        return NASContext(
            search_space=self.search_space,
            evaluator=None,  # Proxy evaluation by default
            dataset=self.dataset,
        )

    def validate_solution(self, solution, instance_data: dict | None = None) -> bool:
        """Validate a NAS solution (architecture)."""
        from src.domains.nas.architecture import NeuralArchitecture
        if not isinstance(solution, NeuralArchitecture):
            return False
        return self.search_space.is_valid(solution)


@dataclass
class NASBenchConfig:
    """Configuration for NAS-Bench-201 benchmark experiments.

    Uses CellArchitecture representation with tabular lookup evaluation.
    """

    dataset: str = "cifar10"
    representation_type: RepresentationType = RepresentationType.ARCHITECTURE_DAG
    nasbench_path: str | None = None
    use_proxy: bool = False
    seed: int | None = None

    def create_context(self, instance_data: dict | None = None) -> "NASBenchContext":
        """Create a NAS-Bench-201 context for evaluation.

        Returns:
            NASBenchContext using CellArchitecture + tabular lookup.
        """
        from src.domains.nas.context import NASBenchContext
        from src.domains.nas.nasbench_evaluator import NASBench201Evaluator

        evaluator = NASBench201Evaluator(
            nasbench_path=self.nasbench_path,
            dataset=self.dataset,
            use_proxy=self.use_proxy,
            seed=self.seed,
        )
        return NASBenchContext(evaluator=evaluator, dataset=self.dataset)

    def validate_solution(self, solution, instance_data: dict | None = None) -> bool:
        """Validate a NAS-Bench-201 solution (CellArchitecture)."""
        from src.domains.nas.cell_architecture import CellArchitecture, NUM_EDGES, NUM_OPS
        if not isinstance(solution, CellArchitecture):
            return False
        if len(solution.edges) != NUM_EDGES:
            return False
        return all(0 <= e < NUM_OPS for e in solution.edges)
