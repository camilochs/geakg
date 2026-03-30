"""Instance Pool with Instance Hardness Sampling.

This module provides multi-instance training support where instances are
sampled based on their hardness (gap from optimal).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domains.base import DomainConfig


# =============================================================================
# Known optimums for TSPLIB instances
# =============================================================================

TSPLIB_OPTIMA: dict[str, float] = {
    "eil51": 426,
    "berlin52": 7542,
    "kroA100": 21282,
    "ch150": 6528,
    "pcb442": 50778,
    "rat783": 8806,
    "pr1002": 259045,
    "u1060": 224094,
    "vm1084": 239297,
    "d1291": 50801,
}


# =============================================================================
# Evaluation Budget
# =============================================================================


@dataclass
class EvaluationBudget:
    """Tracker de presupuesto de evaluaciones.

    Termination is based on max_tokens (LLM token budget).
    Tracks input/output tokens separately for accurate cost calculation.
    """

    max_tokens: int = 500_000  # Total token budget (primary termination condition)
    prompt_tokens: int = 0  # Input tokens used
    completion_tokens: int = 0  # Output tokens used
    fitness_evals: int = 0
    synth_calls: int = 0

    @property
    def tokens_used(self) -> int:
        """Total tokens used (input + output)."""
        return self.prompt_tokens + self.completion_tokens

    def can_continue(self) -> bool:
        """Check if we can continue (token budget not exhausted)."""
        return self.tokens_used < self.max_tokens

    def record_fitness_eval(self, count: int = 1) -> None:
        """Record fitness evaluation(s)."""
        self.fitness_evals += count

    def record_synth_call(self) -> None:
        """Record synthesized synthesis call."""
        self.synth_calls += 1

    def update_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token counts from LLM stats."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def get_stats(self) -> dict[str, Any]:
        """Get budget statistics."""
        return {
            "max_tokens": self.max_tokens,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_remaining": self.max_tokens - self.tokens_used,
            "fitness_evals": self.fitness_evals,
            "synth_calls": self.synth_calls,
        }

    def estimate_cost_usd(self, model: str = "gpt-4o-mini") -> float:
        """Estimate cost in USD based on model pricing.

        Pricing (per 1M tokens):
        - gpt-4o-mini: $0.15 input, $0.60 output
        - gpt-4o: $2.50 input, $10.00 output
        - gpt-5-mini: ~$0.30 input, $1.20 output (estimated)
        - gpt-5.2: $1.75 input, $14.00 output
        """
        # Pricing per 1M tokens (input, output)
        pricing = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-5-mini": (0.30, 1.20),
            "gpt-5": (1.00, 8.00),
            "gpt-5.1": (1.50, 12.00),
            "gpt-5.2": (1.75, 14.00),
            "gpt-5.2-codex": (1.75, 14.00),
        }

        input_price, output_price = pricing.get(model, (0.15, 0.60))

        input_cost = (self.prompt_tokens / 1_000_000) * input_price
        output_cost = (self.completion_tokens / 1_000_000) * output_price

        return input_cost + output_cost


# =============================================================================
# Instance Info
# =============================================================================


@dataclass
class InstanceInfo:
    """Information about a single instance in the pool."""

    instance_id: str
    instance_data: dict[str, Any]  # distance_matrix, dimension, etc.
    optimal: float | None = None
    champion_fitness: float = float("inf")
    champion_gap: float = float("inf")
    selection_probability: float = 1.0
    selection_count: int = 0  # Times this instance was selected

    @property
    def dimension(self) -> int:
        """Get instance dimension."""
        return self.instance_data.get("dimension", 0)


# =============================================================================
# Instance Pool
# =============================================================================


@dataclass
class InstancePool:
    """Pool of instances with Instance Hardness Sampling.

    Instances are sampled based on how difficult they are for the current
    champion (best-so-far algorithm). Harder instances (higher gap) have
    higher probability of being selected.

    The hardness_exponent controls how aggressively we favor difficult instances:
    - 1.0: Linear (default) - probability proportional to gap
    - 2.0: Quadratic - much stronger preference for difficult instances
    - 0.5: Square root - more balanced sampling
    """

    domain: str = "tsp"
    update_frequency: int = 10
    max_probability: float = 0.6  # Cap to avoid single instance dominating
    hardness_exponent: float = 2.0  # Exponent for hardness weighting (higher = more focus on difficult)
    instances: list[InstanceInfo] = field(default_factory=list)
    _champion_path: list[str] | None = None
    _last_update_iteration: int = -1

    def load_instances_from_dir(
        self,
        instances_dir: str | Path,
        limit: int | None = None,
        optimals: dict[str, float] | None = None,
    ) -> None:
        """Load instances from a directory.

        Args:
            instances_dir: Directory containing instance files.
            limit: Maximum number of instances to load.
            optimals: Dictionary mapping instance names to optimal values.
        """
        instances_dir = Path(instances_dir)

        if self.domain == "tsp":
            self._load_tsp_instances(instances_dir, limit, optimals or TSPLIB_OPTIMA)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")

    def _load_tsp_instances(
        self,
        instances_dir: Path,
        limit: int | None,
        optimals: dict[str, float],
    ) -> None:
        """Load TSP instances from directory, sorted by size (smallest first)."""
        from src.domains.tsp import TSPDomain

        tsp_domain = TSPDomain()

        # Load all instances first to get their sizes
        all_instances = []
        for path in instances_dir.glob("*.tsp"):
            try:
                instance = tsp_domain.load_instance(path)
                all_instances.append((path, instance))
            except Exception:
                pass

        # Sort by dimension (smallest first)
        all_instances.sort(key=lambda x: x[1].dimension)

        # Extract sorted file paths
        files = [path for path, _ in all_instances]

        if limit:
            files = files[:limit]

        for path in files:
            try:
                instance = tsp_domain.load_instance(path)
                instance_id = path.stem

                # Get optimal from various sources
                optimal = (
                    instance.optimal_cost or optimals.get(instance_id) or optimals.get(instance.name)
                )

                instance_data = {
                    "distance_matrix": instance.distance_matrix,
                    "dimension": instance.dimension,
                    "name": instance.name,
                }

                self.instances.append(
                    InstanceInfo(
                        instance_id=instance_id,
                        instance_data=instance_data,
                        optimal=optimal,
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        # Initialize uniform probabilities
        if self.instances:
            uniform = 1.0 / len(self.instances)
            for inst in self.instances:
                inst.selection_probability = uniform

    def load_instances_from_files(
        self,
        instance_files: list[str],
        optimals: dict[str, float] | None = None,
    ) -> None:
        """Load specific instance files.

        Args:
            instance_files: List of paths to instance files.
            optimals: Dictionary mapping instance names to optimal values.
        """
        if self.domain == "tsp":
            self._load_tsp_instance_files(instance_files, optimals or TSPLIB_OPTIMA)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")

    def _load_tsp_instance_files(
        self,
        instance_files: list[str],
        optimals: dict[str, float],
    ) -> None:
        """Load specific TSP instance files."""
        from src.domains.tsp import TSPDomain

        tsp_domain = TSPDomain()

        for file_path in instance_files:
            path = Path(file_path)
            try:
                instance = tsp_domain.load_instance(path)
                instance_id = path.stem

                # Get optimal from various sources
                optimal = (
                    instance.optimal_cost or optimals.get(instance_id) or optimals.get(instance.name)
                )

                instance_data = {
                    "distance_matrix": instance.distance_matrix,
                    "dimension": instance.dimension,
                    "name": instance.name,
                }

                self.instances.append(
                    InstanceInfo(
                        instance_id=instance_id,
                        instance_data=instance_data,
                        optimal=optimal,
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        # Sort by dimension (smallest first) for consistency
        self.instances.sort(key=lambda x: x.dimension)

        # Initialize uniform probabilities
        if self.instances:
            uniform = 1.0 / len(self.instances)
            for inst in self.instances:
                inst.selection_probability = uniform

    def sample_instance(self) -> InstanceInfo:
        """Sample an instance based on hardness-weighted probabilities."""
        if not self.instances:
            raise ValueError("No instances loaded")

        probs = [inst.selection_probability for inst in self.instances]
        selected = random.choices(self.instances, weights=probs, k=1)[0]
        selected.selection_count += 1
        return selected

    def should_update_champion(self, iteration: int) -> bool:
        """Check if champion should be updated this iteration."""
        return iteration - self._last_update_iteration >= self.update_frequency

    def update_champion(
        self,
        champion_path: list[str],
        evaluate_fn: Any,
        iteration: int,
        force: bool = False,
    ) -> int:
        """Update champion and recalculate selection probabilities.

        Args:
            champion_path: Operator path of the champion.
            evaluate_fn: Function to evaluate a path on an instance.
                         Signature: (operator_path, instance_data) -> fitness
            iteration: Current iteration number.
            force: Force update even if not due.

        Returns:
            Number of fitness evaluations performed.
        """
        if not force and not self.should_update_champion(iteration):
            return 0

        self._champion_path = champion_path
        self._last_update_iteration = iteration
        evals_performed = 0

        # Evaluate champion on ALL instances
        for inst in self.instances:
            fitness = evaluate_fn(champion_path, inst.instance_data)
            inst.champion_fitness = fitness
            evals_performed += 1

            if inst.optimal and inst.optimal > 0:
                inst.champion_gap = 100 * (fitness - inst.optimal) / inst.optimal
            else:
                # Without optimal, use fitness normalized by dimension
                inst.champion_gap = fitness / max(inst.dimension, 1)

        # Recalculate probabilities
        self._recalculate_probabilities()

        return evals_performed

    def _recalculate_probabilities(self) -> None:
        """Recalculate selection probabilities based on champion gaps.

        Uses log(dimension) normalization to prevent large instances
        from dominating the selection.
        """
        if not self.instances:
            return

        # Normalize gaps by log(dimension) to balance different sizes
        # Then apply hardness exponent to control how aggressively we favor difficult instances
        normalized_gaps = []
        for inst in self.instances:
            dimension = inst.dimension or 100
            # Use max to avoid division by zero
            normalized_gap = inst.champion_gap / max(math.log2(dimension + 1), 1)
            # Apply hardness exponent: higher exponent = more focus on difficult instances
            weighted_gap = max(normalized_gap, 0.001) ** self.hardness_exponent
            normalized_gaps.append(weighted_gap)

        total = sum(normalized_gaps)

        if total > 0:
            for i, inst in enumerate(self.instances):
                prob = normalized_gaps[i] / total
                # Cap probability to avoid domination
                inst.selection_probability = min(prob, self.max_probability)

            # Renormalize after capping
            total_prob = sum(inst.selection_probability for inst in self.instances)
            if total_prob > 0:
                for inst in self.instances:
                    inst.selection_probability /= total_prob
        else:
            # All gaps zero: uniform distribution
            uniform = 1.0 / len(self.instances)
            for inst in self.instances:
                inst.selection_probability = uniform

    def get_aggregate_gap(self) -> float:
        """Get average gap of champion over all instances."""
        gaps = [inst.champion_gap for inst in self.instances if inst.champion_gap < float("inf")]
        return sum(gaps) / len(gaps) if gaps else float("inf")

    def get_weighted_gap(self) -> float:
        """Get weighted average gap, where difficult instances weigh more."""
        weighted_sum = 0.0
        total_weight = 0.0

        for inst in self.instances:
            if inst.champion_gap < float("inf"):
                weight = inst.selection_probability
                weighted_sum += inst.champion_gap * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else float("inf")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics for logging."""
        if not self.instances:
            return {"n_instances": 0}

        valid_instances = [inst for inst in self.instances if inst.champion_gap < float("inf")]

        return {
            "n_instances": len(self.instances),
            "avg_champion_gap": self.get_aggregate_gap(),
            "weighted_champion_gap": self.get_weighted_gap(),
            "hardest_instance": (
                max(valid_instances, key=lambda x: x.champion_gap).instance_id
                if valid_instances
                else None
            ),
            "easiest_instance": (
                min(valid_instances, key=lambda x: x.champion_gap).instance_id
                if valid_instances
                else None
            ),
            "selection_counts": {inst.instance_id: inst.selection_count for inst in self.instances},
            "probabilities": {
                inst.instance_id: round(inst.selection_probability, 4) for inst in self.instances
            },
            "per_instance_gaps": {
                inst.instance_id: round(inst.champion_gap, 2)
                for inst in self.instances
                if inst.champion_gap < float("inf")
            },
        }

    def get_per_instance_gaps(self) -> dict[str, float]:
        """Get gap for each instance."""
        return {
            inst.instance_id: inst.champion_gap
            for inst in self.instances
            if inst.champion_gap < float("inf")
        }

    def get_instance_gaps_for_synthesis(self) -> list[dict]:
        """Get instance gap info formatted for synthesized synthesis.

        Returns:
            List of dicts with name, gap, and size for each instance.
            Can be passed directly to BottleneckReport.instance_gaps after
            converting to InstanceGap objects.
        """
        result = []
        for inst in self.instances:
            if inst.champion_gap < float("inf"):
                result.append({
                    "name": inst.instance_id,
                    "gap": inst.champion_gap / 100.0,  # Convert from % to ratio
                    "size": inst.dimension,
                })
        return result

    def __len__(self) -> int:
        return len(self.instances)
