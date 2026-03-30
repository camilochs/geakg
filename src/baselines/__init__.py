"""Baseline implementations (LLaMEA, GP)."""

from src.baselines.genetic_programming import GPResult, TSPGeneticProgramming
from src.baselines.llamea_wrapper import (
    LLaMEABaseline,
    LLaMEAResult,
    create_tsp_fitness_wrapper,
    create_tsp_task_prompt,
)

__all__ = [
    "LLaMEABaseline",
    "LLaMEAResult",
    "create_tsp_task_prompt",
    "create_tsp_fitness_wrapper",
    "TSPGeneticProgramming",
    "GPResult",
]
