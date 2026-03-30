"""Design-Space Prompting templates for L1 operator generation.

This module provides the prompt template for generating operator variants
using the Design-Space Prompting technique. The LLM must:
1. Use the specified design choices (selection, scope, information, acceptance)
2. Explain interaction effects between choices
3. Generate valid Python code using the ctx protocol

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/  <-- THIS FILE
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/
"""

# =============================================================================
# MAIN DESIGN-SPACE PROMPT
# =============================================================================

DESIGN_SPACE_PROMPT = """You are designing a {role_category} operator for the **{domain}** problem.

**DOMAIN CONTEXT - {domain}:**
- Solutions are permutations representing tour order
- Goal: minimize total distance (sum of consecutive edges + return to start)
- Edge costs are symmetric distances between cities
- The best operators exploit GEOMETRIC STRUCTURE: detecting/removing crossing edges, segment manipulations, exploiting triangle inequality
- Think about what makes tours BAD (crossings, detours) and how to FIX them efficiently

**GOAL: Create a POWERFUL operator using your knowledge of effective {domain} techniques.**

The original operator (A₀) is intentionally simple. You must create something MUCH BETTER by:
- Using proven algorithmic patterns that work well for {domain}
- Exploiting the geometric/distance structure of the problem
- Implementing efficient neighborhood exploration
- Using smart move evaluation (delta calculations, not full re-evaluation)

**Original operator (A₀) - this is WEAK, you must do BETTER:**
```python
{original_code}
```

**Design Space - guide your approach:**
{design_point}

**Context API:**
- `ctx.cost(solution, i)` → cost contribution at index i
- `ctx.delta(solution, "swap", i, j)` → cost change for swapping i,j
- `ctx.delta(solution, "reverse", i, j)` → cost change for reversing segment [i:j+1]
- `ctx.neighbors(solution, i, k)` → k nearest indices to i
- `ctx.evaluate(solution)` → total tour cost
- `ctx.valid(solution)` → True if valid permutation

**Rules:**
1. Copy first: `result = solution[:]`
2. Use `% n` for circular indexing (tour wraps around)
3. Return `result if ctx.valid(result) else solution`
4. Import random/math at function start if needed
5. Use delta calculations for efficiency, not full evaluate() in inner loops

Return JSON:
{{
  "name": "{category}_<descriptive_name>",
  "design_choices": {{"selection": "...", "scope": "...", "information": "...", "acceptance": "..."}},
  "structural_changes": "<explain your algorithmic approach and why it's effective for {domain}>",
  "code": "def {category}_<name>(solution, ctx):\\n    ..."
}}
"""


# =============================================================================
# JSON SCHEMA FOR VALIDATION
# =============================================================================

DESIGN_SPACE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Operator name following pattern: category_descriptive_name",
        },
        "design_choices": {
            "type": "object",
            "properties": {
                "selection": {"type": "string"},
                "scope": {"type": "string"},
                "information": {"type": "string"},
                "acceptance": {"type": "string"},
            },
            "required": ["selection", "scope", "information", "acceptance"],
        },
        "structural_changes": {
            "type": "string",
            "description": "Explanation of how the algorithm differs structurally from A₀",
        },
        "code": {
            "type": "string",
            "description": "Complete Python function code",
        },
    },
    "required": ["name", "design_choices", "structural_changes", "code"],
}


# =============================================================================
# CATEGORY-SPECIFIC GUIDANCE
# =============================================================================

CATEGORY_GUIDANCE = {
    "construction": """**Construction operators** build tours from scratch.
- Good constructions avoid creating crossings from the start
- Consider insertion position carefully (where in the partial tour?)
- Greedy by distance is OK, but regret-based or geometric awareness is better
- Return a complete permutation [0..n-1]""",

    "local_search": """**Local search operators** improve existing tours.
- The BEST local search for TSP exploits edge reversals to remove crossings
- Segment operations (reverse, relocate) are more powerful than point swaps
- Use delta calculations: removing edges (i,i+1) and (j,j+1), adding new edges
- First-improvement is faster, best-improvement finds better local optima
- Consider scanning edges in order of "badness" (long edges, crossing edges)""",

    "perturbation": """**Perturbation operators** escape local optima.
- Good perturbations break the tour structure without destroying all quality
- Segment-based perturbations (cut and reconnect differently) work well
- Random swaps are WEAK; structured perturbations are better
- The perturbation should make moves that local search cannot undo in one step""",
}


def build_design_space_prompt(
    role: str,
    role_category: str,
    original_code: str,
    design_point: dict[str, str],
    domain: str = "TSP",
) -> str:
    """Build the complete prompt for design-space generation.

    Args:
        role: Target role name (e.g., "ls_intensify_small")
        role_category: Category ("construction", "local_search", "perturbation")
        original_code: Base operator code (A₀)
        design_point: Selected design choices
        domain: Target domain (e.g., "TSP", "VRP", "JSSP")

    Returns:
        Formatted prompt string
    """
    from src.geakg.layers.l1.design_space import format_design_point

    # Get category-specific guidance
    guidance = CATEGORY_GUIDANCE.get(role_category, "")

    # Build the main prompt
    prompt = DESIGN_SPACE_PROMPT.format(
        role_category=role_category,
        category=role_category,
        original_code=original_code.strip(),
        design_point=format_design_point(design_point),
        domain=domain.upper(),
    )

    # Add category guidance if available
    if guidance:
        prompt = prompt.replace(
            "**Design Space",
            f"{guidance}\n\n**Design Space",
        )

    return prompt


# =============================================================================
# PROMPT FOR REFINEMENT (fixing errors)
# =============================================================================

REFINEMENT_PROMPT = """Fix the errors in this operator while keeping the design choices.

**Original code with errors:**
```python
{code}
```

**Errors to fix:**
{errors}

**Design choices (preserve these):**
{design_point}

**Context API:**
- ctx.cost(solution, i), ctx.delta(solution, "swap", i, j)
- ctx.neighbors(solution, i, k), ctx.evaluate(solution), ctx.valid(solution)

**Rules:** Copy first, use % n for indexing, validate before return.

Return JSON:
{{
  "code": "def <fixed_function>(solution, ctx):\\n    ...",
  "changes_made": "<describe what was fixed>"
}}
"""


def build_refinement_prompt(
    code: str,
    errors: list[str],
    design_point: dict[str, str],
) -> str:
    """Build prompt for fixing errors in generated code.

    Args:
        code: Code with errors
        errors: List of error messages
        design_point: Original design choices to preserve

    Returns:
        Formatted refinement prompt
    """
    from src.geakg.layers.l1.design_space import format_design_point

    return REFINEMENT_PROMPT.format(
        code=code.strip(),
        errors="\n".join(f"- {e}" for e in errors),
        design_point=format_design_point(design_point),
    )
