"""NAS-Bench-Graph Evaluator: tabular lookup for GNN architecture evaluation.

Wraps the nas_bench_graph API (lightread) to provide instant accuracy
lookups for architectures in the NAS-Bench-Graph search space.

Datasets: cora, citeseer, pubmed, cs, physics, photo, computers, arxiv, proteins
Metric: Accuracy (higher is better, 0-100)

Requires nas_bench_graph package.

Reference: Qin et al., "NAS-Bench-Graph: Benchmarking Graph Neural
Architecture Search", NeurIPS 2022 Datasets and Benchmarks.
"""

from __future__ import annotations

import random
from typing import Any

from loguru import logger

from src.domains.nas.graph_architecture import (
    GraphArchitecture,
    GRAPH_NUM_NODES,
    GRAPH_NUM_OPS,
    GRAPH_OPERATIONS,
    GRAPH_MAX_CONN,
)


# Valid datasets in NAS-Bench-Graph
VALID_GRAPH_DATASETS = {
    "cora", "citeseer", "pubmed",
    "cs", "physics",
    "photo", "computers",
    "arxiv", "proteins",
}


class NASBenchGraphEvaluator:
    """Evaluator using NAS-Bench-Graph tabular benchmark.

    Provides O(1) lookup of test accuracy for architectures in the
    NAS-Bench-Graph search space (26,206 unique architectures).

    Requires nas_bench_graph package.
    """

    def __init__(
        self,
        dataset: str = "cora",
        seed: int | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            dataset: Graph dataset for evaluation.
            seed: Random seed.
        """
        if dataset not in VALID_GRAPH_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Valid: {sorted(VALID_GRAPH_DATASETS)}"
            )

        self._dataset = dataset
        self._cache: dict[tuple, float] = {}
        self._bench = None
        self._eval_count = 0
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._load_benchmark()

    def _load_benchmark(self) -> None:
        """Load NAS-Bench-Graph data for the configured dataset."""
        from nas_bench_graph import light_read
        self._bench = light_read(self._dataset)
        logger.info(
            f"Loaded NAS-Bench-Graph for dataset={self._dataset} "
            f"({len(self._bench)} entries)"
        )

    @property
    def dataset(self) -> str:
        """Current evaluation dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: str) -> None:
        """Change evaluation dataset (reloads benchmark, clears cache)."""
        if value not in VALID_GRAPH_DATASETS:
            raise ValueError(
                f"Unknown dataset '{value}'. "
                f"Valid: {sorted(VALID_GRAPH_DATASETS)}"
            )
        if value != self._dataset:
            self._dataset = value
            self._cache.clear()
            self._bench = None
            self._load_benchmark()
            logger.info(f"Evaluator dataset changed to {self._dataset}")

    @property
    def eval_count(self) -> int:
        """Number of evaluations performed."""
        return self._eval_count

    def evaluate(
        self,
        arch: GraphArchitecture,
        dataset: str | None = None,
    ) -> float:
        """Evaluate architecture, returning test accuracy (0-100).

        Uses O(1) tabular lookup from NAS-Bench-Graph.

        Args:
            arch: Graph architecture to evaluate.
            dataset: Override dataset for this evaluation.

        Returns:
            Test accuracy as percentage (0-100). Higher is better.
        """
        cache_key = (
            tuple(arch.connectivity),
            tuple(arch.operations),
            dataset or self._dataset,
        )

        if cache_key in self._cache:
            self._eval_count += 1
            return self._cache[cache_key]

        accuracy = self._benchmark_evaluate(arch, dataset)

        self._cache[cache_key] = accuracy
        self._eval_count += 1
        return accuracy

    def _benchmark_evaluate(
        self,
        arch: GraphArchitecture,
        dataset: str | None = None,
    ) -> float:
        """Evaluate using NAS-Bench-Graph lookup.

        NAS-Bench-Graph returns accuracy in [0, 1]. We scale to [0, 100]
        for consistency with NAS-Bench-201.

        Returns 0.0 for architectures outside the valid search space
        (e.g., connectivity patterns not in the benchmark).
        """
        bench = self._bench
        if dataset and dataset != self._dataset:
            from nas_bench_graph import light_read
            bench = light_read(dataset)

        try:
            nb_arch = arch.to_arch()
            h = nb_arch.valid_hash()
            info = bench[h]
            # NAS-Bench-Graph stores accuracy in [0, 1], scale to [0, 100]
            accuracy = float(info["perf"]) * 100.0
            return round(accuracy, 2)
        except (UnboundLocalError, KeyError, TypeError):
            # Architecture outside valid search space
            return 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "dataset": self._dataset,
            "eval_count": self._eval_count,
            "cache_size": len(self._cache),
            "total_architectures": 26206,
        }

    def reset_eval_count(self) -> None:
        """Reset the evaluation counter."""
        self._eval_count = 0
