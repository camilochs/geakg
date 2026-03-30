"""Design Space for structured operator variation.

This module defines orthogonal design axes that guide LLM generation.
Instead of free-form generation, the LLM explores a formalized space
of 256 combinations (4 axes x 4 options each).

Design-Space Prompting benefits:
- Forces exploration beyond common templates (GA, SA, TS)
- Produces structural diversity, not superficial variation
- LLM must justify interaction effects between axes
- Enables systematic coverage of the design space
"""

import random
from typing import Any


# =============================================================================
# DESIGN AXES - Orthogonal dimensions for operator design
# =============================================================================

DESIGN_AXES: dict[str, list[str]] = {
    "selection": [
        "greedy (always pick best)",
        "probabilistic (weighted random)",
        "tournament (compare k candidates)",
        "adaptive (change based on progress)",
    ],
    "scope": [
        "point (single element)",
        "segment (consecutive subsequence)",
        "distributed (scattered elements)",
        "global (entire solution)",
    ],
    "information": [
        "cost-driven (use ctx.cost)",
        "delta-driven (use ctx.delta)",
        "neighbor-driven (use ctx.neighbors)",
        "history-driven (track recent changes)",
    ],
    "acceptance": [
        "improving only (delta < 0)",
        "threshold (delta < epsilon)",
        "probabilistic (accept worse with p)",
        "best-seen (track global best)",
    ],
}


# Total combinations: 4 * 4 * 4 * 4 = 256
TOTAL_COMBINATIONS = 1
for options in DESIGN_AXES.values():
    TOTAL_COMBINATIONS *= len(options)


def sample_design_point(rng: random.Random) -> dict[str, str]:
    """Sample a random point from the design space.

    Args:
        rng: Random number generator for reproducibility

    Returns:
        Dictionary mapping each axis to a selected option
    """
    return {
        axis: rng.choice(options)
        for axis, options in DESIGN_AXES.items()
    }


def format_design_point(point: dict[str, str]) -> str:
    """Format a design point for inclusion in prompts.

    Args:
        point: Design point dictionary

    Returns:
        Formatted string with one axis per line
    """
    return "\n".join(f"- {axis}: {choice}" for axis, choice in point.items())


def design_point_to_key(point: dict[str, str]) -> tuple:
    """Convert design point to hashable key for deduplication.

    Args:
        point: Design point dictionary

    Returns:
        Tuple that can be used as dict key or in set
    """
    return tuple(sorted(point.items()))


def enumerate_all_points() -> list[dict[str, str]]:
    """Generate all 256 possible design points.

    Useful for systematic exploration rather than random sampling.

    Returns:
        List of all design point combinations
    """
    from itertools import product

    axes = list(DESIGN_AXES.keys())
    all_options = [DESIGN_AXES[axis] for axis in axes]

    points = []
    for combo in product(*all_options):
        point = dict(zip(axes, combo))
        points.append(point)

    return points


def get_complementary_point(point: dict[str, str]) -> dict[str, str]:
    """Get a design point that differs maximally from the given one.

    Useful for ensuring diversity when sampling multiple variants.

    Args:
        point: Original design point

    Returns:
        Design point with different selection on each axis
    """
    complementary = {}
    for axis, current in point.items():
        options = DESIGN_AXES[axis]
        # Pick an option different from current
        other_options = [opt for opt in options if opt != current]
        if other_options:
            complementary[axis] = other_options[0]  # Take first different
        else:
            complementary[axis] = current
    return complementary


def describe_design_space() -> str:
    """Get human-readable description of the design space.

    Returns:
        Formatted description of all axes and options
    """
    lines = ["Design Space for Operator Generation", "=" * 40, ""]

    for axis, options in DESIGN_AXES.items():
        lines.append(f"## {axis.upper()}")
        for i, opt in enumerate(options, 1):
            lines.append(f"  {i}. {opt}")
        lines.append("")

    lines.append(f"Total combinations: {TOTAL_COMBINATIONS}")
    return "\n".join(lines)


# =============================================================================
# CATEGORY-SPECIFIC CONSTRAINTS
# =============================================================================

# Some combinations make more sense for certain operator categories
CATEGORY_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    "construction": {
        # Construction typically builds from scratch, so "improving only" is less relevant
        "acceptance": [
            "greedy (always pick best)",
            "probabilistic (weighted random)",
            "threshold (delta < epsilon)",
        ],
    },
    "local_search": {
        # Local search focuses on improvement
        "acceptance": DESIGN_AXES["acceptance"],  # All options valid
    },
    "perturbation": {
        # Perturbation accepts changes to escape local optima
        "acceptance": [
            "probabilistic (accept worse with p)",
            "threshold (delta < epsilon)",
            "best-seen (track global best)",
        ],
    },
}


def sample_design_point_for_category(
    rng: random.Random,
    category: str,
) -> dict[str, str]:
    """Sample a design point respecting category constraints.

    Args:
        rng: Random number generator
        category: "construction", "local_search", or "perturbation"

    Returns:
        Design point dictionary
    """
    constraints = CATEGORY_CONSTRAINTS.get(category, {})

    point = {}
    for axis, options in DESIGN_AXES.items():
        # Use constrained options if available for this category
        available = constraints.get(axis, options)
        point[axis] = rng.choice(available)

    return point
