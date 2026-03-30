"""NAS-Bench-201 Evaluator: O(1) tabular lookup for architecture evaluation.

Wraps the nats_bench API to provide instant accuracy lookups for any
architecture in the NAS-Bench-201 search space.

Supports three datasets:
- cifar10: CIFAR-10 (test accuracy, 200 epochs)
- cifar100: CIFAR-100 (test accuracy, 200 epochs)
- ImageNet16-120: ImageNet-16-120 (test accuracy, 200 epochs)

Reference: Dong & Yang, "NAS-Bench-201: Extending the Scope of Reproducible
Neural Architecture Search", ICLR 2020.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from loguru import logger

from src.domains.nas.cell_architecture import CellArchitecture, OPERATIONS, NUM_OPS


# Valid dataset names
VALID_DATASETS = {"cifar10", "cifar100", "ImageNet16-120"}

# Dataset name mapping (user-friendly -> NAS-Bench-201 API names)
DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "imagenet16": "ImageNet16-120",
    "imagenet16-120": "ImageNet16-120",
    "imagenet": "ImageNet16-120",
}


class NASBench201Evaluator:
    """Evaluator using NAS-Bench-201 tabular benchmark.

    Provides O(1) lookup of test accuracy for any architecture in the
    5^6 = 15,625 architecture search space.

    Requires nats_bench package and NATS-tss-v1_0-3ffb9-simple data.
    """

    def __init__(
        self,
        nasbench_path: str | Path | None = None,
        dataset: str = "cifar10",
        seed: int | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            nasbench_path: Path to NAS-Bench-201 data file (NATS-tss-v1_0-3ffb9-simple).
                          If None, reads from NASBENCH_PATH env var.
            dataset: Dataset for evaluation (cifar10, cifar100, ImageNet16-120).
            seed: Random seed.
        """
        self._dataset = self._resolve_dataset(dataset)
        self._cache: dict[str, float] = {}
        self._api = None
        self._eval_count = 0
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._load_benchmark(nasbench_path)

    @staticmethod
    def _resolve_dataset(dataset: str) -> str:
        """Resolve dataset name to NAS-Bench-201 format."""
        normalized = dataset.lower().strip()
        if normalized in DATASET_ALIASES:
            return DATASET_ALIASES[normalized]
        if dataset in VALID_DATASETS:
            return dataset
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Valid options: {sorted(VALID_DATASETS | set(DATASET_ALIASES.keys()))}"
        )

    def _load_benchmark(self, nasbench_path: str | Path | None) -> None:
        """Load NAS-Bench-201 data."""
        if nasbench_path is None:
            nasbench_path = os.environ.get("NASBENCH_PATH")

        if nasbench_path is None:
            raise RuntimeError(
                "NAS-Bench-201 path not set. "
                "Set NASBENCH_PATH env var or pass nasbench_path."
            )

        nasbench_path = Path(nasbench_path)
        if not nasbench_path.exists():
            raise FileNotFoundError(
                f"NAS-Bench-201 data not found at {nasbench_path}."
            )

        from nats_bench import create
        self._api = create(str(nasbench_path), "tss", fast_mode=True, verbose=False)
        logger.info(
            f"Loaded NAS-Bench-201 ({len(self._api)} architectures) "
            f"for dataset={self._dataset}"
        )

    @property
    def dataset(self) -> str:
        """Current evaluation dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: str) -> None:
        """Change evaluation dataset (clears cache)."""
        new_dataset = self._resolve_dataset(value)
        if new_dataset != self._dataset:
            self._dataset = new_dataset
            self._cache.clear()
            logger.info(f"Evaluator dataset changed to {self._dataset}")

    @property
    def eval_count(self) -> int:
        """Number of evaluations performed."""
        return self._eval_count

    def evaluate(
        self,
        arch: CellArchitecture,
        dataset: str | None = None,
        hp: str = "200",
    ) -> float:
        """Evaluate architecture, returning test accuracy (0-100).

        Uses O(1) tabular lookup from NAS-Bench-201.

        Args:
            arch: Cell architecture to evaluate.
            dataset: Override dataset for this evaluation.
            hp: Training epochs (NAS-Bench-201 supports "12" and "200").

        Returns:
            Test accuracy as percentage (0-100). Higher is better.
        """
        ds = self._resolve_dataset(dataset) if dataset else self._dataset

        # Check cache
        cache_key = f"{arch.to_nasbench_string()}|{ds}|{hp}"
        if cache_key in self._cache:
            self._eval_count += 1
            return self._cache[cache_key]

        accuracy = self._benchmark_evaluate(arch, ds, hp)

        self._cache[cache_key] = accuracy
        self._eval_count += 1
        return accuracy

    def _benchmark_evaluate(
        self,
        arch: CellArchitecture,
        dataset: str,
        hp: str,
    ) -> float:
        """Evaluate using NAS-Bench-201 lookup."""
        arch_str = arch.to_nasbench_string()

        # NAS-Bench-201 API: query by architecture string
        index = self._api.query_index_by_arch(arch_str)
        if index < 0:
            raise ValueError(f"Architecture not found in NAS-Bench-201: {arch_str}")

        # Get test accuracy for the dataset
        info = self._api.get_more_info(index, dataset, hp=hp, is_random=False)
        accuracy = info["test-accuracy"]
        return accuracy

    def get_optimal(self, dataset: str | None = None) -> tuple[CellArchitecture, float]:
        """Find the optimal architecture by exhaustive search.

        Args:
            dataset: Dataset to evaluate on.

        Returns:
            Tuple of (best architecture, best accuracy).
        """
        ds = self._resolve_dataset(dataset) if dataset else self._dataset
        best_arch = None
        best_acc = -1.0

        for idx in range(NUM_OPS ** 6):
            arch = CellArchitecture.from_index(idx)
            acc = self.evaluate(arch, ds)
            if acc > best_acc:
                best_acc = acc
                best_arch = arch

        return best_arch, best_acc

    def get_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "dataset": self._dataset,
            "eval_count": self._eval_count,
            "cache_size": len(self._cache),
            "total_architectures": NUM_OPS ** 6,
        }

    def reset_eval_count(self) -> None:
        """Reset the evaluation counter."""
        self._eval_count = 0
