"""Base operators (A₀) for each role.

These are simple, validated, functional operators that serve as
the starting point for AFO (Always-From-Original) generation.

Each operator:
- Compiles and executes without errors
- Uses only the ctx protocol (cost, delta, neighbors, evaluate, valid)
- Is simple but functional
- Handles edge cases (wrap-around with %, empty checks)
"""

# All 11 roles in the metagraph
ALL_ROLES = [
    # Construction (4 roles)
    "const_greedy",
    "const_insertion",
    "const_savings",
    "const_random",
    # Local Search (4 roles)
    "ls_intensify_small",
    "ls_intensify_medium",
    "ls_intensify_large",
    "ls_chain",
    # Perturbation (3 roles)
    "pert_escape_small",
    "pert_escape_large",
    "pert_adaptive",
]

# Base operators by role - GENERIC operators using only ctx.evaluate()
#
# These operators are DOMAIN-AGNOSTIC and work with any permutation problem.
# They only use:
#   - ctx.evaluate(solution) -> float  (cost of a solution)
#   - len(solution) -> int             (problem size)
#
# NO domain-specific methods like ctx.delta(), ctx.cost(), ctx.instance, etc.
#
BASE_OPERATORS = {
    # ==========================================================================
    # CONSTRUCTION - build initial solutions
    # ==========================================================================
    "const_greedy": '''
def const_greedy(solution, ctx):
    """Greedy construction: try multiple random starts, keep best."""
    import random
    n = len(solution)
    best_result, best_cost = None, float('inf')
    for _ in range(min(n, 5)):
        result = list(range(n))
        random.shuffle(result)
        # Greedy improvement: swap if better
        for i in range(n):
            for j in range(i + 1, n):
                result[i], result[j] = result[j], result[i]
                cost = ctx.evaluate(result)
                if cost >= best_cost:
                    result[i], result[j] = result[j], result[i]  # Undo
        cost = ctx.evaluate(result)
        if cost < best_cost:
            best_cost, best_result = cost, result[:]
    return best_result if best_result else list(range(n))
''',

    "const_insertion": '''
def const_insertion(solution, ctx):
    """Insertion construction: build solution by inserting elements at best position."""
    n = len(solution)
    result = [0]
    remaining = list(range(1, n))
    while remaining:
        best_city, best_pos, best_cost = None, 0, float('inf')
        for city in remaining:
            for pos in range(len(result) + 1):
                candidate = result[:pos] + [city] + result[pos:]
                cost = ctx.evaluate(candidate)
                if cost < best_cost:
                    best_cost, best_city, best_pos = cost, city, pos
        result.insert(best_pos, best_city)
        remaining.remove(best_city)
    return result
''',

    "const_savings": '''
def const_savings(solution, ctx):
    """Savings construction: insert element that minimizes cost increase."""
    n = len(solution)
    result = [0]
    remaining = list(range(1, n))
    while remaining:
        best_city, best_pos, best_delta = None, 0, float('inf')
        current_cost = ctx.evaluate(result) if len(result) > 1 else 0
        for city in remaining:
            for pos in range(len(result) + 1):
                candidate = result[:pos] + [city] + result[pos:]
                delta = ctx.evaluate(candidate) - current_cost
                if delta < best_delta:
                    best_delta, best_city, best_pos = delta, city, pos
        result.insert(best_pos, best_city)
        remaining.remove(best_city)
    return result
''',

    "const_random": '''
def const_random(solution, ctx):
    """Random construction: shuffle the solution."""
    import random
    result = list(range(len(solution)))
    random.shuffle(result)
    return result
''',

    # ==========================================================================
    # LOCAL SEARCH - improve existing solution using only ctx.evaluate()
    # ==========================================================================
    "ls_intensify_small": '''
def ls_intensify_small(solution, ctx):
    """First-improvement swaps using evaluate()."""
    result = solution[:]
    n = len(result)
    current_cost = ctx.evaluate(result)
    for _ in range(50):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                result[i], result[j] = result[j], result[i]
                new_cost = ctx.evaluate(result)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                result[i], result[j] = result[j], result[i]  # Undo
            if improved:
                break
        if not improved:
            break
    return result
''',

    "ls_intensify_medium": '''
def ls_intensify_medium(solution, ctx):
    """2-opt: reverse segments using evaluate()."""
    result = solution[:]
    n = len(result)
    current_cost = ctx.evaluate(result)
    for _ in range(50):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # 2-opt: reverse segment [i+1, j]
                result[i+1:j+1] = result[i+1:j+1][::-1]
                new_cost = ctx.evaluate(result)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                result[i+1:j+1] = result[i+1:j+1][::-1]  # Undo
            if improved:
                break
        if not improved:
            break
    return result
''',

    "ls_intensify_large": '''
def ls_intensify_large(solution, ctx):
    """Best-improvement swaps using evaluate()."""
    result = solution[:]
    n = len(result)
    for _ in range(50):
        current_cost = ctx.evaluate(result)
        best_delta, best_i, best_j = 0, -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                result[i], result[j] = result[j], result[i]
                new_cost = ctx.evaluate(result)
                delta = new_cost - current_cost
                if delta < best_delta:
                    best_delta, best_i, best_j = delta, i, j
                result[i], result[j] = result[j], result[i]  # Undo
        if best_i >= 0:
            result[best_i], result[best_j] = result[best_j], result[best_i]
        else:
            break
    return result
''',

    "ls_chain": '''
def ls_chain(solution, ctx):
    """Chain: first-improvement swaps then 2-opt."""
    result = solution[:]
    n = len(result)
    # Phase 1: First-improvement swaps
    current_cost = ctx.evaluate(result)
    for i in range(n):
        for j in range(i + 1, n):
            result[i], result[j] = result[j], result[i]
            new_cost = ctx.evaluate(result)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                result[i], result[j] = result[j], result[i]  # Undo
    # Phase 2: 2-opt
    for i in range(n - 1):
        for j in range(i + 2, n):
            result[i+1:j+1] = result[i+1:j+1][::-1]
            new_cost = ctx.evaluate(result)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                result[i+1:j+1] = result[i+1:j+1][::-1]  # Undo
    return result
''',

    # ==========================================================================
    # PERTURBATION - escape local optima (don't need evaluate, just perturb)
    # ==========================================================================
    "pert_escape_small": '''
def pert_escape_small(solution, ctx):
    """Swap 2-3 random pairs."""
    import random
    result = solution[:]
    n = len(result)
    for _ in range(random.randint(2, 3)):
        i, j = random.sample(range(n), 2)
        result[i], result[j] = result[j], result[i]
    return result
''',

    "pert_escape_large": '''
def pert_escape_large(solution, ctx):
    """Shuffle a random segment (25% of solution)."""
    import random
    result = solution[:]
    n = len(result)
    size = max(2, n // 4)
    start = random.randint(0, n - size)
    segment = result[start:start + size]
    random.shuffle(segment)
    result[start:start + size] = segment
    return result
''',

    "pert_adaptive": '''
def pert_adaptive(solution, ctx):
    """Adaptive perturbation: shuffle worst segment based on cost."""
    import random
    result = solution[:]
    n = len(result)
    # Perturb a random segment (simpler, domain-agnostic version)
    size = max(2, n // 5)
    start = random.randint(0, n - size)
    segment = result[start:start + size]
    random.shuffle(segment)
    result[start:start + size] = segment
    return result
''',
}


def get_base_operator(role: str) -> str:
    """Get the base operator code for a role.

    Args:
        role: Role name (e.g., "ls_intensify_small")

    Returns:
        Python code string for the base operator

    Raises:
        KeyError: If role is not found
    """
    if role not in BASE_OPERATORS:
        raise KeyError(f"Unknown role: {role}. Available: {ALL_ROLES}")
    return BASE_OPERATORS[role].strip()


def get_role_category(role: str, role_schema=None) -> str:
    """Get category from role name.

    Args:
        role: Role name (e.g., "ls_intensify_small" or "topo_feedforward")
        role_schema: Optional RoleSchema for non-optimization roles.

    Returns:
        Category string (e.g., "construction", "topology")
    """
    # Try schema first
    if role_schema is not None:
        try:
            return role_schema.get_role_category(role)
        except KeyError:
            pass

    # Fallback: prefix-based for optimization roles
    if role.startswith("const_"):
        return "construction"
    elif role.startswith("ls_"):
        return "local_search"
    elif role.startswith("pert_"):
        return "perturbation"
    return "local_search"  # Default
