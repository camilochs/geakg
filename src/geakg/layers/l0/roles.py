"""Abstract Roles for Domain-Agnostic Algorithm Composition.

This module defines the 11 Abstract Roles that form the core vocabulary
for expressing optimization strategies independently of any specific domain.

The key insight: LLMs induce a grammar of optimization strategies, not specific
algorithms. Transfer occurs through principles (roles), not artifacts (operators).

Taxonomy (3 categories, 11 roles):
- CONSTRUCTION (4): Build initial solutions
- LOCAL_SEARCH (4): Intensify around current solution
- PERTURBATION (3): Escape local optima
"""

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field


class AbstractRole(str, Enum):
    """The 11 abstract roles for domain-agnostic algorithm composition.

    These roles capture semantic behavior (what the operator does conceptually)
    rather than implementation (how it does it in a specific domain).

    3 categories, 11 roles total:
    - CONSTRUCTION (4): const_greedy, const_insertion, const_savings, const_random
    - LOCAL_SEARCH (4): ls_intensify_small, ls_intensify_medium, ls_intensify_large, ls_chain
    - PERTURBATION (3): pert_escape_small, pert_escape_large, pert_adaptive
    """

    # === CONSTRUCTION: Build initial solutions ===
    CONST_GREEDY = "const_greedy"
    """Greedy construction: build solution by locally optimal choices.
    Examples: Nearest Neighbor (TSP), SPT dispatch (JSSP), First Fit (BP)."""

    CONST_INSERTION = "const_insertion"
    """Insertion-based construction: iteratively insert elements.
    Examples: Cheapest/Farthest Insertion (TSP), EST insertion (JSSP)."""

    CONST_SAVINGS = "const_savings"
    """Savings/merging construction: combine partial solutions.
    Examples: Clarke-Wright Savings (VRP), Christofides (TSP)."""

    CONST_RANDOM = "const_random"
    """Random/diverse construction: introduce randomness for diversity.
    Examples: Random permutation, Random insertion."""

    # === LOCAL SEARCH: Intensify around current solution ===
    LS_INTENSIFY_SMALL = "ls_intensify_small"
    """Small neighborhood local search: simple, fast moves.
    Examples: 2-opt, swap, insert (TSP); adjacent swap (JSSP)."""

    LS_INTENSIFY_MEDIUM = "ls_intensify_medium"
    """Medium neighborhood local search: more complex moves.
    Examples: 3-opt, Or-opt (TSP); block moves (JSSP)."""

    LS_INTENSIFY_LARGE = "ls_intensify_large"
    """Large neighborhood local search: expensive but thorough.
    Examples: Lin-Kernighan (TSP); ejection chains (JSSP)."""

    LS_CHAIN = "ls_chain"
    """Chained/VND local search: systematic neighborhood exploration.
    Examples: Variable Neighborhood Descent, sequential application."""

    # === PERTURBATION: Escape local optima ===
    PERT_ESCAPE_SMALL = "pert_escape_small"
    """Mild perturbation: small disruption to escape.
    Examples: Double bridge (TSP); random swap (JSSP)."""

    PERT_ESCAPE_LARGE = "pert_escape_large"
    """Strong perturbation: significant restructuring.
    Examples: Ruin-recreate, LNS (both domains)."""

    PERT_ADAPTIVE = "pert_adaptive"
    """Adaptive perturbation: history/context-based escape.
    Examples: Guided mutation, reactive perturbation."""


class RoleCategory(str, Enum):
    """Categories of abstract roles (3 categories)."""

    CONSTRUCTION = "construction"
    LOCAL_SEARCH = "local_search"
    PERTURBATION = "perturbation"


class RoleNode(BaseModel):
    """A node in the meta-graph representing an abstract role.

    Contains metadata that helps the LLM and MMAS make informed decisions.
    Accepts both AbstractRole and plain str for role ID (generalization).
    """

    role: Union[str, AbstractRole]
    description: str = Field(..., description="Human-readable description")
    category: RoleCategory = Field(..., description="Role category")
    expected_cost: str = Field(
        default="O(n²)",
        description="Expected computational complexity"
    )
    exploration_bias: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0=pure intensification, 1=pure exploration"
    )

    def is_construction(self) -> bool:
        return self.category == RoleCategory.CONSTRUCTION

    def is_local_search(self) -> bool:
        return self.category == RoleCategory.LOCAL_SEARCH

    def is_perturbation(self) -> bool:
        return self.category == RoleCategory.PERTURBATION



# =============================================================================
# ROLE_CATALOG: Complete metadata for all 11 roles
# =============================================================================

ROLE_CATALOG: dict[AbstractRole, dict[str, Any]] = {
    # === CONSTRUCTION ===
    AbstractRole.CONST_GREEDY: {
        "description": "Build solution by making locally optimal choices at each step",
        "category": RoleCategory.CONSTRUCTION,
        "expected_cost": "O(n²)",
        "exploration_bias": 0.1,
        "typical_quality": "Good initial solutions, may miss global optimum",
        "when_to_use": "Starting point for most algorithms",
        "examples_tsp": ["greedy_nearest_neighbor", "nearest_addition"],
        "examples_jssp": ["spt_dispatch", "mwkr_dispatch", "fifo_dispatch"],
        "examples_bp": ["first_fit", "best_fit"],
    },
    AbstractRole.CONST_INSERTION: {
        "description": "Build solution by iteratively inserting elements at best positions",
        "category": RoleCategory.CONSTRUCTION,
        "expected_cost": "O(n²) to O(n³)",
        "exploration_bias": 0.2,
        "typical_quality": "Often better than greedy, more expensive",
        "when_to_use": "When quality matters more than speed",
        "examples_tsp": ["cheapest_insertion", "farthest_insertion", "nearest_insertion"],
        "examples_jssp": ["est_insertion", "ect_insertion"],
        "examples_bp": ["best_fit_decreasing"],
    },
    AbstractRole.CONST_SAVINGS: {
        "description": "Build solution by combining partial solutions based on savings",
        "category": RoleCategory.CONSTRUCTION,
        "expected_cost": "O(n² log n)",
        "exploration_bias": 0.15,
        "typical_quality": "High quality for routing problems",
        "when_to_use": "VRP-like problems, when merging is natural",
        "examples_tsp": ["savings_heuristic", "christofides_construction"],
        "examples_jssp": ["shifting_bottleneck"],
        "examples_bp": ["first_fit_decreasing"],
    },
    AbstractRole.CONST_RANDOM: {
        "description": "Build solution with random choices for diversity",
        "category": RoleCategory.CONSTRUCTION,
        "expected_cost": "O(n)",
        "exploration_bias": 1.0,
        "typical_quality": "Low quality, high diversity",
        "when_to_use": "Population initialization, restart diversification",
        "examples_tsp": ["random_insertion", "random_permutation"],
        "examples_jssp": ["random_dispatch"],
        "examples_bp": ["random_fit"],
    },

    # === LOCAL SEARCH ===
    AbstractRole.LS_INTENSIFY_SMALL: {
        "description": "Improve solution via small, fast neighborhood moves",
        "category": RoleCategory.LOCAL_SEARCH,
        "expected_cost": "O(n²)",
        "exploration_bias": 0.1,
        "typical_quality": "Quick improvements, may plateau",
        "when_to_use": "First improvement phase, quick refinement",
        "examples_tsp": ["two_opt", "swap", "insert", "invert"],
        "examples_jssp": ["adjacent_swap", "critical_swap"],
        "examples_bp": ["item_swap", "item_move"],
    },
    AbstractRole.LS_INTENSIFY_MEDIUM: {
        "description": "Improve solution via medium-complexity moves",
        "category": RoleCategory.LOCAL_SEARCH,
        "expected_cost": "O(n³)",
        "exploration_bias": 0.2,
        "typical_quality": "Better solutions, slower convergence",
        "when_to_use": "After small moves plateau, moderate effort",
        "examples_tsp": ["three_opt", "or_opt", "relocate"],
        "examples_jssp": ["block_move", "block_swap"],
        "examples_bp": ["multi_item_move"],
    },
    AbstractRole.LS_INTENSIFY_LARGE: {
        "description": "Improve solution via expensive, thorough moves",
        "category": RoleCategory.LOCAL_SEARCH,
        "expected_cost": "O(n³) to O(n⁴)",
        "exploration_bias": 0.25,
        "typical_quality": "Near-optimal for local region",
        "when_to_use": "Final polishing, when quality critical",
        "examples_tsp": ["lin_kernighan", "lk_search"],
        "examples_jssp": ["ejection_chain", "path_relinking"],
        "examples_bp": ["repack_subset"],
    },
    AbstractRole.LS_CHAIN: {
        "description": "Systematically explore multiple neighborhoods",
        "category": RoleCategory.LOCAL_SEARCH,
        "expected_cost": "Variable",
        "exploration_bias": 0.3,
        "typical_quality": "Robust, combines multiple moves",
        "when_to_use": "When single moves insufficient",
        "examples_tsp": ["variable_neighborhood", "vnd"],
        "examples_jssp": ["vns_local", "sequential_vnd"],
        "examples_bp": ["vnd_bins"],
    },

    # === PERTURBATION ===
    AbstractRole.PERT_ESCAPE_SMALL: {
        "description": "Mild perturbation to escape local optimum",
        "category": RoleCategory.PERTURBATION,
        "expected_cost": "O(n)",
        "exploration_bias": 0.6,
        "typical_quality": "Maintains structure, small disruption",
        "when_to_use": "Early stagnation, controlled escape",
        "examples_tsp": ["double_bridge", "random_segment_shuffle"],
        "examples_jssp": ["random_delay", "critical_block_shuffle"],
        "examples_bp": ["remove_reinsert"],
    },
    AbstractRole.PERT_ESCAPE_LARGE: {
        "description": "Strong perturbation for significant restructuring",
        "category": RoleCategory.PERTURBATION,
        "expected_cost": "O(n) to O(n²)",
        "exploration_bias": 0.85,
        "typical_quality": "Major disruption, new search region",
        "when_to_use": "Deep stagnation, need fresh start",
        "examples_tsp": ["ruin_recreate", "large_neighborhood_search"],
        "examples_jssp": ["ruin_recreate", "destroy_repair"],
        "examples_bp": ["empty_bins_repack"],
    },
    AbstractRole.PERT_ADAPTIVE: {
        "description": "Context-aware perturbation based on search history",
        "category": RoleCategory.PERTURBATION,
        "expected_cost": "O(n)",
        "exploration_bias": 0.7,
        "typical_quality": "Intelligent escape, learns from history",
        "when_to_use": "Long runs, when patterns emerge",
        "examples_tsp": ["guided_mutation", "adaptive_mutation"],
        "examples_jssp": ["guided_perturbation", "frequency_based_shake"],
        "examples_bp": ["adaptive_destroy"],
    },
}


def get_role_node(role: AbstractRole) -> RoleNode:
    """Get a RoleNode for the given abstract role."""
    info = ROLE_CATALOG[role]
    return RoleNode(
        role=role,
        description=info["description"],
        category=info["category"],
        expected_cost=info["expected_cost"],
        exploration_bias=info["exploration_bias"],
    )


def get_all_role_nodes() -> list[RoleNode]:
    """Get RoleNodes for all 11 abstract roles."""
    return [get_role_node(role) for role in AbstractRole]


def get_roles_by_category(category: RoleCategory) -> list[AbstractRole]:
    """Get all roles in a specific category."""
    return [
        role for role, info in ROLE_CATALOG.items()
        if info["category"] == category
    ]


def get_construction_roles() -> list[AbstractRole]:
    """Get all construction roles."""
    return get_roles_by_category(RoleCategory.CONSTRUCTION)


def get_local_search_roles() -> list[AbstractRole]:
    """Get all local search roles."""
    return get_roles_by_category(RoleCategory.LOCAL_SEARCH)


def get_perturbation_roles() -> list[AbstractRole]:
    """Get all perturbation roles."""
    return get_roles_by_category(RoleCategory.PERTURBATION)




def get_role_description_for_llm() -> str:
    """Generate a description of all roles for LLM prompts.

    This is used in the MetaGraphGenerator to explain the role taxonomy
    to the LLM so it can design meta-algorithms.
    """
    lines = [
        "# Abstract Roles for Algorithm Composition",
        "",
        "These roles represent semantic behaviors, not specific implementations.",
        "Design a meta-algorithm by specifying which roles connect to which.",
        "",
    ]

    for category in RoleCategory:
        lines.append(f"## {category.value.upper()}")
        roles = get_roles_by_category(category)
        for role in roles:
            info = ROLE_CATALOG[role]
            lines.append(f"- **{role.value}**: {info['description']}")
            lines.append(f"  - Cost: {info['expected_cost']}, Exploration: {info['exploration_bias']:.1f}")
            lines.append(f"  - When: {info['when_to_use']}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Role Transition Rules (domain-agnostic constraints)
# =============================================================================

# Valid transitions between role categories (3 categories only)
VALID_CATEGORY_TRANSITIONS: dict[RoleCategory, list[RoleCategory]] = {
    RoleCategory.CONSTRUCTION: [
        RoleCategory.LOCAL_SEARCH,
        RoleCategory.PERTURBATION,  # For restart strategies
    ],
    RoleCategory.LOCAL_SEARCH: [
        RoleCategory.LOCAL_SEARCH,  # Chain of LS
        RoleCategory.PERTURBATION,  # Escape
    ],
    RoleCategory.PERTURBATION: [
        RoleCategory.LOCAL_SEARCH,  # Re-optimize after perturbation
        RoleCategory.CONSTRUCTION,  # Full restart
        RoleCategory.PERTURBATION,  # Chain perturbations
    ],
}


def is_valid_role_transition(source: AbstractRole, target: AbstractRole) -> bool:
    """Check if a transition between two roles is valid.

    This enforces domain-agnostic constraints on algorithm composition.
    """
    source_cat = ROLE_CATALOG[source]["category"]
    target_cat = ROLE_CATALOG[target]["category"]

    # Same role is always valid (for repetition)
    if source == target:
        return True

    return target_cat in VALID_CATEGORY_TRANSITIONS.get(source_cat, [])


def get_valid_next_roles(current: AbstractRole) -> list[AbstractRole]:
    """Get all valid roles that can follow the current role."""
    current_cat = ROLE_CATALOG[current]["category"]
    valid_cats = VALID_CATEGORY_TRANSITIONS.get(current_cat, [])

    return [
        role for role in AbstractRole
        if ROLE_CATALOG[role]["category"] in valid_cats or role == current
    ]
