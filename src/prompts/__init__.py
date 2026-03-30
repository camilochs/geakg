"""Synthesis prompt templates.

This module provides domain-agnostic templates for operator synthesis.
Each domain (TSP, JSSP, etc.) fills in domain-specific placeholders.
"""

# =============================================================================
# CATEGORY GUIDANCE - Concise guidance per category
# =============================================================================

CATEGORY_GUIDANCE = {
    "construction": """Build a complete solution from scratch. Aim for good initial quality.""",

    "local_search": """Improve the solution by exploring its neighborhood. Aim for consistent improvement.""",

    "perturbation": """Escape local optima to explore new regions of the search space.""",
}


# =============================================================================
# ALGORITHMIC BUILDING BLOCKS - Not used (kept for reference)
# =============================================================================

ALGORITHMIC_BUILDING_BLOCKS = ""  # Removed to keep prompts generic


# =============================================================================
# QUALITY CHECKLIST
# =============================================================================

QUALITY_CHECKLIST = """Return valid solutions. Handle edge cases. Keep complexity reasonable."""


# =============================================================================
# ANTI-PATTERNS
# =============================================================================

ANTI_PATTERNS = """AVOID: O(n³) loops, modifying input directly, invalid returns, indexing beyond list bounds (use i % n for wrap-around)."""


# =============================================================================
# MAIN TEMPLATE
# =============================================================================

SYNTHESIS_TEMPLATE = """You are an expert in combinatorial optimization. Design a novel and effective {role_category} operator.

{category_guidance}

**Role:** {role} - {role_description}
**Problem:** {domain_name} (n={instance_size_info})

{bottleneck_context}

{domain_specific_guidance}

{synthesis_history_section}

**Do NOT repeat these:** {techniques_already_tried}

{previous_errors}

**Signature:** {function_signature}
{safety_constraints}

Return JSON: {{"operator_name": "{category}_<name>", "code": "def {category}_<name>(solution, ctx):\\n    ...", "reasoning": "..."}}"""


# =============================================================================
# HISTORY SECTIONS
# =============================================================================

SYNTHESIS_HISTORY_SECTION = """**What worked before:** {long_term_reflections}
{suggested_combinations}"""

NO_HISTORY_SECTION = ""




# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_role_category(role: str) -> str:
    """Get the category of a role based on its prefix.

    Categories:
    - construction: const_* roles (4 roles)
    - local_search: ls_* roles (4 roles)
    - perturbation: pert_* roles (3 roles)

    Total: 11 roles across 3 categories.
    """
    if role.startswith("const_"):
        return "construction"
    elif role.startswith("ls_"):
        return "local_search"
    elif role.startswith("pert_"):
        return "perturbation"
    return "local_search"  # Default


def build_synthesis_prompt(
    domain_name: str,
    role: str,
    domain_specific_guidance: str = "",
    long_term_reflections: str = "",
    suggested_combinations: str = "",
    instance_size: int = 0,
    **kwargs
) -> str:
    """Build synthesis prompt with domain and category-specific parts.

    Args:
        domain_name: Name of the domain (e.g., "TSP", "JSSP")
        role: The target role (e.g., "ls_intensify_small")
        domain_specific_guidance: Optional domain-specific hints
        long_term_reflections: Accumulated insights from previous synthesis
        suggested_combinations: LLM-suggested technique combinations
        instance_size: Average problem size for scalability hints
        **kwargs: Template placeholders (role_description, etc.)

    Returns:
        Formatted prompt string
    """
    category = get_role_category(role)
    category_guidance = CATEGORY_GUIDANCE.get(category, "")

    # Build the history section based on whether we have history
    has_history = bool(long_term_reflections.strip()) or bool(suggested_combinations.strip())

    if has_history:
        synthesis_history_section = SYNTHESIS_HISTORY_SECTION.format(
            long_term_reflections=long_term_reflections or "",
            suggested_combinations=suggested_combinations or "",
        )
    else:
        synthesis_history_section = NO_HISTORY_SECTION

    # Format instance size info
    instance_size_info = str(instance_size) if instance_size > 0 else "variable"

    return SYNTHESIS_TEMPLATE.format(
        domain_name=domain_name,
        role=role,
        role_category=category.replace("_", " "),
        category=category,
        category_guidance=category_guidance,
        domain_specific_guidance=domain_specific_guidance,
        synthesis_history_section=synthesis_history_section,
        instance_size_info=instance_size_info,
        **kwargs
    )




# =============================================================================
# DOMAIN-AGNOSTIC OPERATORS - For Cross-Domain Transfer Learning
# =============================================================================
# Inspired by Butler Lampson's "Hints for Computer System Design":
# - "An interface should capture the minimum essentials of an abstraction"
# - "Keep secrets of the implementation"
# =============================================================================

AGNOSTIC_OPERATOR_SIGNATURE = """
def {name}(solution: list, ctx) -> list:
    '''Domain-agnostic operator for {role_category}.

    This operator works on TSP, VRP, JSSP without modification.
    The context (ctx) hides all domain-specific details.

    Args:
        solution: Current solution as list (e.g., [0, 3, 1, 2, 4] for n=5)
        ctx: DomainContext with these methods:

            ctx.cost(solution, i) -> float
                Returns cost contribution of element at index i.
                Example: ctx.cost(solution, 0) returns cost of first element.

            ctx.delta(solution, move_type, i, j) -> float
                Returns change in cost if move applied (without executing).
                move_type is "swap", "insert", or "reverse".
                Example: ctx.delta(solution, "swap", 2, 5) returns delta for swapping indices 2 and 5.

            ctx.neighbors(solution, i, k) -> list[int]
                Returns k indices most related to element at index i.
                IMPORTANT: Requires both i (index) and k (count) arguments.
                Example: ctx.neighbors(solution, 3, 5) returns 5 neighbors of index 3.

            ctx.evaluate(solution) -> float
                Returns total solution cost.
                Example: total = ctx.evaluate(solution)

            ctx.valid(solution) -> bool
                Returns True if solution satisfies all constraints.
                Example: if ctx.valid(new_solution): return new_solution

    Returns:
        Modified solution (same length, same elements, different order)
    '''
"""

AGNOSTIC_OPERATOR_GUIDANCE = """
## DOMAIN-AGNOSTIC RULES

Following Lampson's principle "Keep secrets": the context HIDES domain details.
Your operator MUST use ctx methods, NEVER domain-specific data.

### CORRECT USAGE EXAMPLES:

```python
import random

def local_search_example(solution: list, ctx) -> list:
    '''Example showing correct ctx usage.'''
    n = len(solution)
    result = solution.copy()  # Always copy first!

    # 1. Find worst element by cost
    worst_idx = max(range(n), key=lambda i: ctx.cost(result, i))

    # 2. Get 5 neighbors of the worst element (BOTH arguments required!)
    neighbor_indices = ctx.neighbors(result, worst_idx, 5)

    # 3. Try swapping with each neighbor
    for j in neighbor_indices:
        # Check delta BEFORE applying
        delta = ctx.delta(result, "swap", worst_idx, j)
        if delta < 0:  # Improving move
            result[worst_idx], result[j] = result[j], result[worst_idx]
            break

    # 4. Validate and return
    return result if ctx.valid(result) else solution


def perturbation_example(solution: list, ctx) -> list:
    '''Example perturbation operator.'''
    n = len(solution)
    if n < 4:
        return solution

    result = solution.copy()

    # Shuffle a random segment
    start = random.randint(0, n - 3)
    end = random.randint(start + 2, min(start + n // 4, n))
    segment = result[start:end]
    random.shuffle(segment)
    result[start:end] = segment

    return result if ctx.valid(result) else solution
```

### COMMON MISTAKES (DO NOT DO THIS):

```python
# WRONG: Missing arguments to ctx.neighbors()
neighbors = ctx.neighbors()           # ERROR! Needs (solution, i, k)
neighbors = ctx.neighbors(solution)   # ERROR! Needs i and k

# CORRECT:
neighbors = ctx.neighbors(solution, i, 5)  # Get 5 neighbors of index i

# WRONG: Accessing domain-specific data
distance_matrix[i][j]   # NO! Use ctx.delta() or ctx.cost()
ctx.distances           # NO! Not part of the interface
ctx.distance_matrix     # NO! Hidden by context

# CORRECT: Use only the 5 ctx methods
cost = ctx.cost(solution, i)
delta = ctx.delta(solution, "swap", i, j)
neighbors = ctx.neighbors(solution, i, k)
total = ctx.evaluate(solution)
ok = ctx.valid(solution)
```

### THE 5 CONTEXT METHODS (MEMORIZE THESE):

| Method | Arguments | Returns | Purpose |
|--------|-----------|---------|---------|
| ctx.cost(solution, i) | solution, index | float | Cost at index i |
| ctx.delta(solution, move, i, j) | solution, "swap"/"insert"/"reverse", i, j | float | Cost change |
| ctx.neighbors(solution, i, k) | solution, index, count | list[int] | k nearest indices |
| ctx.evaluate(solution) | solution | float | Total cost |
| ctx.valid(solution) | solution | bool | Feasibility check |

### WHY THIS MATTERS:

Your operator will run on TSP, VRP, JSSP, BPP without modification.
The ctx adapts to each domain - same code works everywhere.
"""

AGNOSTIC_OPERATOR_TEMPLATE = """You are synthesizing a DOMAIN-AGNOSTIC operator.

{agnostic_signature}

{agnostic_guidance}

## YOUR TASK: Create a {role_category} operator

Role: {role}
Description: {role_description}

{category_guidance}

## REQUIREMENTS:

1. Use ONLY ctx.cost(), ctx.delta(), ctx.neighbors(), ctx.evaluate(), ctx.valid()
2. DO NOT reference any domain-specific variables
3. Return the modified solution (or original if invalid)
4. Keep implementation simple and efficient

{synthesis_history_section}

Write ONLY the Python function. No explanations.
"""


def build_agnostic_operator_prompt(
    role: str,
    role_description: str = "",
    long_term_reflections: str = "",
    suggested_combinations: str = "",
    **kwargs
) -> str:
    """Build prompt for domain-agnostic operator synthesis.

    These operators use DomainContext protocol and can transfer across domains.

    Args:
        role: Target role (e.g., "ls_intensify_small")
        role_description: Description of what the role should do
        long_term_reflections: Accumulated insights
        suggested_combinations: Technique suggestions
        **kwargs: Additional placeholders

    Returns:
        Formatted prompt for agnostic operator synthesis
    """
    category = get_role_category(role)
    category_guidance = CATEGORY_GUIDANCE.get(category, "")

    # Build history section
    has_history = bool(long_term_reflections.strip()) or bool(suggested_combinations.strip())
    if has_history:
        synthesis_history_section = SYNTHESIS_HISTORY_SECTION.format(
            long_term_reflections=long_term_reflections or "No patterns recorded yet.",
            suggested_combinations=suggested_combinations or "Explore freely.",
        )
    else:
        synthesis_history_section = NO_HISTORY_SECTION

    return AGNOSTIC_OPERATOR_TEMPLATE.format(
        agnostic_signature=AGNOSTIC_OPERATOR_SIGNATURE.format(
            name=f"{category}_{role.replace('_', '')}",
            role_category=category.replace("_", " "),
        ),
        agnostic_guidance=AGNOSTIC_OPERATOR_GUIDANCE,
        role=role,
        role_description=role_description or f"A {category} operator",
        role_category=category.replace("_", " "),
        category_guidance=category_guidance,
        synthesis_history_section=synthesis_history_section,
        **kwargs
    )


# Backward compatibility aliases
L4_SYNTHESIS_TEMPLATE = SYNTHESIS_TEMPLATE
build_l4_prompt = build_synthesis_prompt

__all__ = [
    "SYNTHESIS_TEMPLATE",
    "build_synthesis_prompt",
    "build_agnostic_operator_prompt",
    "get_role_category",
    "CATEGORY_GUIDANCE",
    "ALGORITHMIC_BUILDING_BLOCKS",
    "ANTI_PATTERNS",
    "SYNTHESIS_HISTORY_SECTION",
    "NO_HISTORY_SECTION",
    "AGNOSTIC_OPERATOR_SIGNATURE",
    "AGNOSTIC_OPERATOR_GUIDANCE",
    "AGNOSTIC_OPERATOR_TEMPLATE",
    # Backward compatibility
    "L4_SYNTHESIS_TEMPLATE",
    "build_l4_prompt",
]
