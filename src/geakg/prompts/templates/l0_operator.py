"""L0 Operator Synthesis Templates.

Parametrized templates for generating operators across all families.
Each template is customized based on:
- Family: defines available operations and solution structure
- Domain: provides specific context about the problem
- Role: specifies the expected behavior (construction, LS, perturbation)

The design axes guide the LLM toward diverse operator implementations.
"""

from __future__ import annotations

from src.geakg.contexts.base import OptimizationFamily


# =============================================================================
# FAMILY DESCRIPTIONS
# =============================================================================

FAMILY_DESCRIPTIONS: dict[OptimizationFamily, str] = {
    OptimizationFamily.PERMUTATION: """Solutions are orderings of elements [0, 1, ..., n-1].
Each element appears exactly once. Examples: TSP tours, job schedules.
Key operations: swap (exchange positions), insert (relocate element),
reverse (2-opt, flip segment), or-opt (move segment).""",
    OptimizationFamily.BINARY: """Solutions are bit vectors [0, 1, 0, 1, ...] of length n.
Each bit indicates selection (1) or not (0). Examples: Knapsack items, feature selection.
Key operations: flip (toggle bit), swap 0<->1 (exchange selected/unselected),
multi-flip (toggle multiple bits).""",
    OptimizationFamily.CONTINUOUS: """Solutions are real-valued vectors [x1, x2, ..., xn].
Each variable is bounded within specified limits. Examples: function optimization.
Key operations: perturb (add noise), gradient step, crossover (blend parents),
differential mutation.""",
    OptimizationFamily.PARTITION: """Solutions assign n items to groups [g0, g1, ..., gn-1].
Each item belongs to exactly one group. Examples: bin packing, clustering.
Key operations: move (reassign item), swap (exchange group assignments),
merge groups, split groups.""",
}


# =============================================================================
# CONTEXT METHODS BY FAMILY
# =============================================================================

CONTEXT_METHODS: dict[OptimizationFamily, str] = {
    OptimizationFamily.PERMUTATION: """## Available Context Methods

```python
# Solution manipulation
ctx.swap(solution, i, j) -> list[int]    # Swap elements at positions i, j
ctx.insert(solution, i, j) -> list[int]  # Remove from i, insert at j
ctx.reverse(solution, i, j) -> list[int] # Reverse segment [i, j] (2-opt)
ctx.or_opt(solution, i, length, j) -> list[int]  # Move segment

# Cost evaluation
ctx.evaluate(solution) -> float          # Total solution cost
ctx.cost(solution, i) -> float           # Cost contribution at position i
ctx.delta_swap(solution, i, j) -> float  # Cost change for swap (O(1))
ctx.delta_reverse(solution, i, j) -> float # Cost change for 2-opt (O(1))

# Utilities
ctx.valid(solution) -> bool              # Check validity
ctx.dimension -> int                     # Number of elements
ctx.random_solution() -> list[int]       # Generate random permutation
ctx.copy(solution) -> list[int]          # Deep copy
```""",
    OptimizationFamily.BINARY: """## Available Context Methods

```python
# Solution manipulation
ctx.flip(solution, i) -> list[int]              # Toggle bit at position i
ctx.flip_multiple(solution, indices) -> list[int]  # Toggle multiple bits
ctx.set_bit(solution, i, value) -> list[int]    # Set bit to 0 or 1
ctx.swap_values(solution, i, j) -> list[int]    # Swap values at i, j

# Cost evaluation
ctx.evaluate(solution) -> float          # Total solution cost
ctx.delta_flip(solution, i) -> float     # Cost change for flip

# Utilities
ctx.valid(solution) -> bool              # Check validity (constraints)
ctx.get_selected_indices(solution) -> list[int]   # Indices where bit=1
ctx.get_unselected_indices(solution) -> list[int] # Indices where bit=0
ctx.count_ones(solution) -> int          # Number of selected items
ctx.repair_greedy(solution) -> list[int] # Fix infeasible solution
ctx.dimension -> int                     # Number of bits
ctx.random_solution() -> list[int]       # Random valid solution
```""",
    OptimizationFamily.CONTINUOUS: """## Available Context Methods

```python
# Solution manipulation
ctx.perturb(solution, sigma, indices=None) -> list[float]  # Gaussian noise
ctx.perturb_uniform(solution, delta, indices=None) -> list[float]  # Uniform noise
ctx.gradient_step(solution, gradient, step_size) -> list[float]  # Gradient descent
ctx.clip(solution) -> list[float]        # Clip to bounds
ctx.crossover_blend(p1, p2, alpha) -> list[float]  # BLX-alpha crossover
ctx.differential_mutation(base, d1, d2, F) -> list[float]  # DE mutation

# Cost evaluation
ctx.evaluate(solution) -> float          # Function value (minimize)
ctx.gradient(solution) -> list[float]    # Numerical gradient

# Utilities
ctx.valid(solution) -> bool              # Check within bounds
ctx.bounds -> list[tuple[float, float]]  # (lower, upper) per dimension
ctx.in_bounds(solution) -> bool          # All values within bounds
ctx.dimension -> int                     # Number of variables
ctx.random_solution() -> list[float]     # Random within bounds
ctx.euclidean_distance(s1, s2) -> float  # L2 distance
```""",
    OptimizationFamily.PARTITION: """## Available Context Methods

```python
# Solution manipulation
ctx.move(solution, item, to_group) -> list[int]   # Move item to group
ctx.swap_items(solution, i, j) -> list[int]       # Swap group assignments
ctx.merge_groups(solution, g1, g2) -> list[int]   # Merge g2 into g1
ctx.split_group(solution, group, new_group) -> list[int]  # Split group
ctx.compact_groups(solution) -> list[int]         # Renumber groups 0,1,2,...

# Cost evaluation
ctx.evaluate(solution) -> float          # Total cost
ctx.delta_move(solution, item, to_group) -> float  # Cost change for move

# Group analysis
ctx.get_groups(solution) -> dict[int, list[int]]  # group_id -> items
ctx.group_sizes(solution) -> dict[int, int]       # group_id -> count
ctx.group_load(solution, group) -> float          # Load of a group
ctx.balance_metric(solution) -> float             # Imbalance measure
ctx.num_active_groups(solution) -> int            # Non-empty groups

# Utilities
ctx.valid(solution) -> bool              # Check constraints
ctx.n_items -> int                       # Number of items
ctx.n_groups -> int                      # Maximum groups allowed
ctx.random_solution() -> list[int]       # Random partition
```""",
}


# =============================================================================
# DESIGN AXES (guide diversity in operator synthesis)
# =============================================================================

DESIGN_AXES: dict[str, dict[str, list[str]]] = {
    "construction": {
        "strategy": [
            "greedy (locally optimal choices)",
            "random (uniform sampling)",
            "hybrid (GRASP-style RCL)",
            "hierarchical (build clusters first)",
        ],
        "ordering": [
            "by benefit/cost ratio",
            "by index (deterministic)",
            "by random shuffling",
            "by distance/similarity",
        ],
        "initialization": [
            "empty solution",
            "seed with best element",
            "multiple starting points",
        ],
    },
    "local_search": {
        "neighborhood": [
            "small (single element change)",
            "medium (segment/block operations)",
            "large (multiple elements)",
            "variable (adaptive size)",
        ],
        "selection": [
            "first improvement",
            "best improvement",
            "probabilistic acceptance",
            "restricted candidate list",
        ],
        "termination": [
            "no improvement found",
            "iteration limit",
            "time limit",
            "quality threshold",
        ],
    },
    "perturbation": {
        "intensity": [
            "mild (1-5% change)",
            "moderate (10-20% change)",
            "strong (30%+ change)",
            "adaptive (based on stagnation)",
        ],
        "guidance": [
            "random (uniform selection)",
            "cost-guided (target worst elements)",
            "history-guided (avoid recent moves)",
            "diversity-guided (maximize change)",
        ],
        "structure": [
            "point-wise (independent changes)",
            "block-wise (contiguous segments)",
            "pattern-based (specific structures)",
        ],
    },
}


# =============================================================================
# MAIN L0 TEMPLATE
# =============================================================================

L0_TEMPLATE = """You are designing operators for {family} optimization.

## Family: {family}
{family_description}

## Domain: {domain}
{domain_description}

## Role: {role}
{role_description}

{context_methods}

## Design Axes for {category}
{design_axes}

## Requirements
1. Function signature: `def {operator_name}(solution, ctx) -> solution`
2. Always return a valid solution (use ctx.valid() to check)
3. Use ctx methods, do NOT access internal data structures
4. Keep complexity reasonable (avoid O(n³) loops)
5. Handle edge cases (empty, single element, etc.)

## Previous Operators (do not repeat these patterns)
{previous_operators}

## Output Format
Return ONLY a Python function. No explanations.

```python
def {operator_name}(solution, ctx):
    '''
    {role} operator for {domain}.
    '''
    # Your implementation here
    ...
```"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_role_category(role: str) -> str:
    """Get category from role name."""
    if role.startswith("const_"):
        return "construction"
    elif role.startswith("ls_"):
        return "local_search"
    elif role.startswith("pert_"):
        return "perturbation"
    return "local_search"


def build_design_space_section(category: str) -> str:
    """Build design axes section for a category."""
    axes = DESIGN_AXES.get(category, {})
    if not axes:
        return ""

    lines = []
    for axis_name, options in axes.items():
        lines.append(f"- **{axis_name}**: {', '.join(options)}")

    return "\n".join(lines)


def build_l0_prompt(
    family: OptimizationFamily,
    domain: str,
    role: str,
    role_description: str = "",
    domain_description: str = "",
    previous_operators: list[str] | None = None,
    operator_name: str | None = None,
) -> str:
    """Build complete L0 synthesis prompt.

    Args:
        family: Optimization family
        domain: Domain name (e.g., 'tsp')
        role: Role name (e.g., 'ls_intensify_small')
        role_description: Description of what the role should do
        domain_description: Optional domain-specific context
        previous_operators: Names of operators already generated
        operator_name: Optional custom operator name

    Returns:
        Complete prompt string
    """
    category = get_role_category(role)

    if operator_name is None:
        operator_name = f"{category}_{role.replace('_', '')}"

    family_description = FAMILY_DESCRIPTIONS.get(
        family, "Unknown optimization family."
    )
    context_methods = CONTEXT_METHODS.get(family, "")
    design_axes = build_design_space_section(category)

    prev_ops_str = (
        ", ".join(previous_operators)
        if previous_operators
        else "None yet."
    )

    return L0_TEMPLATE.format(
        family=family.value,
        family_description=family_description,
        domain=domain,
        domain_description=domain_description or f"Optimize {domain} instances.",
        role=role,
        role_description=role_description or f"A {category} operator.",
        context_methods=context_methods,
        category=category,
        design_axes=design_axes,
        operator_name=operator_name,
        previous_operators=prev_ops_str,
    )


__all__ = [
    "L0_TEMPLATE",
    "FAMILY_DESCRIPTIONS",
    "CONTEXT_METHODS",
    "DESIGN_AXES",
    "build_l0_prompt",
    "build_design_space_section",
    "get_role_category",
]
