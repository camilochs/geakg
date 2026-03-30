"""Base operators (A₀) for NAS-Bench-201 cell architecture roles.

These are the starting points for LLM-based L1 operator synthesis.
Each A₀ is a Python code string that the LLM receives as context
to generate more sophisticated variants.

Each operator:
- Compiles and executes without errors
- Takes (solution, ctx) -> solution signature
- Works on CellArchitecture (6 edges, 5 operations)
- Is simple but functional
- Handles edge cases

18 roles × 1 base operator each = 18 operators.
"""

# All 18 NAS roles (same as NASRoleSchema)
ALL_NAS_BENCH_ROLES = [
    "topo_feedforward", "topo_residual", "topo_recursive", "topo_cell_based",
    "act_standard", "act_modern", "act_parametric", "act_mixed",
    "train_optimizer", "train_schedule", "train_augmentation", "train_loss",
    "reg_dropout", "reg_normalization", "reg_weight_decay", "reg_structural",
    "eval_proxy", "eval_full",
]

# Operation indices for reference in generated code
# NONE=0, SKIP_CONNECT=1, NOR_CONV_1X1=2, NOR_CONV_3X3=3, AVG_POOL_3X3=4
# NUM_OPS=5, NUM_EDGES=6

NAS_BENCH_BASE_OPERATORS: dict[str, str] = {
    # ==========================================================================
    # TOPOLOGY — modify cell edge structure
    # ==========================================================================
    "topo_feedforward": '''
def topo_feedforward(solution, ctx):
    """Set a random edge to nor_conv_3x3 (add deep convolution)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 5)
    result.edges[idx] = 3  # nor_conv_3x3
    return result
''',

    "topo_residual": '''
def topo_residual(solution, ctx):
    """Set a random edge to skip_connect (add residual path)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 5)
    result.edges[idx] = 1  # skip_connect
    return result
''',

    "topo_recursive": '''
def topo_recursive(solution, ctx):
    """Swap two random edges to reorganize the cell."""
    import random
    result = solution.copy()
    i = random.randint(0, 5)
    j = random.randint(0, 5)
    while j == i:
        j = random.randint(0, 5)
    result.edges[i], result.edges[j] = result.edges[j], result.edges[i]
    return result
''',

    "topo_cell_based": '''
def topo_cell_based(solution, ctx):
    """Randomize all 6 edges for complete cell redesign."""
    import random
    result = solution.copy()
    for i in range(6):
        result.edges[i] = random.randint(0, 4)
    return result
''',

    # ==========================================================================
    # ACTIVATION — modify edges with semantic bias
    # ==========================================================================
    "act_standard": '''
def act_standard(solution, ctx):
    """Replace a none edge with avg_pool_3x3 (standard activation)."""
    import random
    result = solution.copy()
    none_edges = [i for i, e in enumerate(result.edges) if e == 0]
    if none_edges:
        idx = random.choice(none_edges)
        result.edges[idx] = 4  # avg_pool_3x3
    else:
        idx = random.randint(0, 5)
        result.edges[idx] = 4
    return result
''',

    "act_modern": '''
def act_modern(solution, ctx):
    """Change a conv edge to nor_conv_1x1 (lightweight operation)."""
    import random
    result = solution.copy()
    conv_edges = [i for i, e in enumerate(result.edges) if e == 3]
    if conv_edges:
        idx = random.choice(conv_edges)
        result.edges[idx] = 2  # nor_conv_1x1
    else:
        idx = random.randint(0, 5)
        result.edges[idx] = 2
    return result
''',

    "act_parametric": '''
def act_parametric(solution, ctx):
    """Replace skip_connect with nor_conv_3x3 (add learnable params)."""
    import random
    result = solution.copy()
    skip_edges = [i for i, e in enumerate(result.edges) if e == 1]
    if skip_edges:
        idx = random.choice(skip_edges)
        result.edges[idx] = 3  # nor_conv_3x3
    else:
        idx = random.randint(0, 5)
        result.edges[idx] = 3
    return result
''',

    "act_mixed": '''
def act_mixed(solution, ctx):
    """Randomize 2 edges for mixed exploration."""
    import random
    result = solution.copy()
    indices = random.sample(range(6), 2)
    for idx in indices:
        result.edges[idx] = random.randint(0, 4)
    return result
''',

    # ==========================================================================
    # TRAINING — incremental modifications
    # ==========================================================================
    "train_optimizer": '''
def train_optimizer(solution, ctx):
    """Replace a none edge with a convolution (more trainable ops)."""
    import random
    result = solution.copy()
    none_edges = [i for i, e in enumerate(result.edges) if e == 0]
    if none_edges:
        idx = random.choice(none_edges)
        result.edges[idx] = random.choice([2, 3])  # conv_1x1 or conv_3x3
    return result
''',

    "train_schedule": '''
def train_schedule(solution, ctx):
    """Swap nor_conv_3x3 <-> nor_conv_1x1 (adjust capacity)."""
    import random
    result = solution.copy()
    conv3 = [i for i, e in enumerate(result.edges) if e == 3]
    conv1 = [i for i, e in enumerate(result.edges) if e == 2]
    if conv3 and random.random() < 0.5:
        idx = random.choice(conv3)
        result.edges[idx] = 2
    elif conv1:
        idx = random.choice(conv1)
        result.edges[idx] = 3
    return result
''',

    "train_augmentation": '''
def train_augmentation(solution, ctx):
    """Swap 2 random edges (minor perturbation)."""
    import random
    result = solution.copy()
    i = random.randint(0, 5)
    j = random.randint(0, 5)
    while j == i:
        j = random.randint(0, 5)
    result.edges[i], result.edges[j] = result.edges[j], result.edges[i]
    return result
''',

    "train_loss": '''
def train_loss(solution, ctx):
    """Change an edge to neighbor operation (+/-1 in op list)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 5)
    current = result.edges[idx]
    if random.random() < 0.5:
        result.edges[idx] = (current + 1) % 5
    else:
        result.edges[idx] = (current - 1) % 5
    return result
''',

    # ==========================================================================
    # REGULARIZATION — simplify/prune
    # ==========================================================================
    "reg_dropout": '''
def reg_dropout(solution, ctx):
    """Set a non-none edge to none (drop path)."""
    import random
    result = solution.copy()
    non_none = [i for i, e in enumerate(result.edges) if e != 0]
    if non_none:
        idx = random.choice(non_none)
        result.edges[idx] = 0  # none
    return result
''',

    "reg_normalization": '''
def reg_normalization(solution, ctx):
    """Replace avg_pool with nor_conv_1x1 (normalized op)."""
    import random
    result = solution.copy()
    pool_edges = [i for i, e in enumerate(result.edges) if e == 4]
    if pool_edges:
        idx = random.choice(pool_edges)
        result.edges[idx] = 2  # nor_conv_1x1
    return result
''',

    "reg_weight_decay": '''
def reg_weight_decay(solution, ctx):
    """Replace nor_conv_3x3 with skip_connect (reduce params)."""
    import random
    result = solution.copy()
    conv3 = [i for i, e in enumerate(result.edges) if e == 3]
    if conv3:
        idx = random.choice(conv3)
        result.edges[idx] = 1  # skip_connect
    return result
''',

    "reg_structural": '''
def reg_structural(solution, ctx):
    """Enforce minimum 4 non-none edges (guarantee connectivity)."""
    import random
    result = solution.copy()
    non_none_count = sum(1 for e in result.edges if e != 0)
    while non_none_count < 4:
        none_edges = [i for i, e in enumerate(result.edges) if e == 0]
        if not none_edges:
            break
        idx = random.choice(none_edges)
        result.edges[idx] = random.choice([1, 2, 3])
        non_none_count += 1
    return result
''',

    # ==========================================================================
    # EVALUATION — validate/evaluate
    # ==========================================================================
    "eval_proxy": '''
def eval_proxy(solution, ctx):
    """Validate cell encoding: clamp edges to [0, 4]."""
    result = solution.copy()
    for i in range(6):
        result.edges[i] = max(0, min(4, result.edges[i]))
    return result
''',

    "eval_full": '''
def eval_full(solution, ctx):
    """Trigger full NAS-Bench-201 evaluation via context."""
    result = solution.copy()
    if hasattr(ctx, 'evaluate'):
        _ = ctx.evaluate(result)
    return result
''',
}


def get_nas_bench_base_operator(role: str) -> str:
    """Get the base operator code for a NAS-Bench-201 role.

    Args:
        role: NAS role name (e.g., "topo_feedforward")

    Returns:
        Python code string for the base operator.

    Raises:
        KeyError: If role is not found.
    """
    if role not in NAS_BENCH_BASE_OPERATORS:
        raise KeyError(
            f"Unknown NAS-Bench role: {role}. Available: {ALL_NAS_BENCH_ROLES}"
        )
    return NAS_BENCH_BASE_OPERATORS[role].strip()
