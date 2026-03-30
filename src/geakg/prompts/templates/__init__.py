"""Prompt templates for multi-family operator synthesis.

Templates are parametrized by:
- Family (permutation, binary, continuous, partition)
- Domain (tsp, knapsack, etc.)
- Role (const_greedy, ls_first, pert_random, etc.)
"""

from src.geakg.prompts.templates.l0_operator import (
    L0_TEMPLATE,
    FAMILY_DESCRIPTIONS,
    CONTEXT_METHODS,
    DESIGN_AXES,
    build_l0_prompt,
    build_design_space_section,
)

__all__ = [
    "L0_TEMPLATE",
    "FAMILY_DESCRIPTIONS",
    "CONTEXT_METHODS",
    "DESIGN_AXES",
    "build_l0_prompt",
    "build_design_space_section",
]
