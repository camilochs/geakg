"""Base operators (A₀) for NAS-Bench-Graph GNN architecture roles.

These are the starting points for LLM-based L1 operator synthesis.
Each A₀ is a Python code string that the LLM receives as context
to generate more sophisticated variants.

Each operator:
- Compiles and executes without errors
- Takes (solution, ctx) -> solution signature
- Works on GraphArchitecture (4 connectivity [0,3] + 4 operations [0,8])
- Is simple but functional
- Handles edge cases

Operations: {gcn=0, gat=1, sage=2, gin=3, cheb=4, arma=5, k-gnn=6, identity=7, fc=8}
Connectivity: [0,3] (0=input node, 1-3=prior computing node)

18 roles x 1 base operator each = 18 operators.
"""

# All 18 NAS roles (same as NASRoleSchema — shared across NAS benchmarks)
ALL_NAS_GRAPH_ROLES = [
    "topo_feedforward", "topo_residual", "topo_recursive", "topo_cell_based",
    "act_standard", "act_modern", "act_parametric", "act_mixed",
    "train_optimizer", "train_schedule", "train_augmentation", "train_loss",
    "reg_dropout", "reg_normalization", "reg_weight_decay", "reg_structural",
    "eval_proxy", "eval_full",
]

# Operation indices for reference in generated code
# GCN=0, GAT=1, SAGE=2, GIN=3, CHEB=4, ARMA=5, KGNN=6, IDENTITY=7, FC=8
# NUM_OPS=9, NUM_NODES=4, MAX_CONN=4 (connectivity [0,3])

NAS_GRAPH_BASE_OPERATORS: dict[str, str] = {
    # ==========================================================================
    # TOPOLOGY — modify connectivity (DAG structure)
    # ==========================================================================
    "topo_feedforward": '''
def topo_feedforward(solution, ctx):
    """Set all connectivity to input node (star topology)."""
    result = solution.copy()
    for i in range(4):
        result.connectivity[i] = 0  # All from input
    return result
''',

    "topo_residual": '''
def topo_residual(solution, ctx):
    """Set connectivity to chain pattern (sequential)."""
    result = solution.copy()
    for i in range(4):
        result.connectivity[i] = min(i, 3)
    return result
''',

    "topo_recursive": '''
def topo_recursive(solution, ctx):
    """Swap two random connectivity values."""
    import random
    result = solution.copy()
    i = random.randint(0, 3)
    j = random.randint(0, 3)
    while j == i:
        j = random.randint(0, 3)
    result.connectivity[i], result.connectivity[j] = result.connectivity[j], result.connectivity[i]
    return result
''',

    "topo_cell_based": '''
def topo_cell_based(solution, ctx):
    """Randomize all 4 connectivity values for complete DAG redesign."""
    import random
    result = solution.copy()
    for i in range(4):
        result.connectivity[i] = random.randint(0, 3)
    return result
''',

    # ==========================================================================
    # ACTIVATION — modify operations with GNN-layer bias
    # ==========================================================================
    "act_standard": '''
def act_standard(solution, ctx):
    """Set a random node to identity (skip connection)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 3)
    result.operations[idx] = 7  # identity
    return result
''',

    "act_modern": '''
def act_modern(solution, ctx):
    """Set a random node to GCN (standard message passing)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 3)
    result.operations[idx] = 0  # gcn
    return result
''',

    "act_parametric": '''
def act_parametric(solution, ctx):
    """Set a random node to FC (fully connected layer)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 3)
    result.operations[idx] = 8  # fc
    return result
''',

    "act_mixed": '''
def act_mixed(solution, ctx):
    """Randomize 2 nodes with diverse GNN operations."""
    import random
    result = solution.copy()
    indices = random.sample(range(4), 2)
    gnn_ops = [0, 1, 2, 3, 4, 5]  # gcn, gat, sage, gin, cheb, arma
    for idx in indices:
        result.operations[idx] = random.choice(gnn_ops)
    return result
''',

    # ==========================================================================
    # TRAINING — incremental modifications
    # ==========================================================================
    "train_optimizer": '''
def train_optimizer(solution, ctx):
    """Replace identity/fc with GCN (upgrade to GNN layer)."""
    import random
    result = solution.copy()
    non_gnn = [i for i, o in enumerate(result.operations) if o in (7, 8)]
    if non_gnn:
        idx = random.choice(non_gnn)
        result.operations[idx] = 0  # gcn
    return result
''',

    "train_schedule": '''
def train_schedule(solution, ctx):
    """Swap GCN <-> GAT (attention trade-off)."""
    import random
    result = solution.copy()
    gcn_nodes = [i for i, o in enumerate(result.operations) if o == 0]
    gat_nodes = [i for i, o in enumerate(result.operations) if o == 1]
    if gcn_nodes and random.random() < 0.5:
        idx = random.choice(gcn_nodes)
        result.operations[idx] = 1  # gat
    elif gat_nodes:
        idx = random.choice(gat_nodes)
        result.operations[idx] = 0  # gcn
    return result
''',

    "train_augmentation": '''
def train_augmentation(solution, ctx):
    """Swap 2 random nodes (both connectivity and operations)."""
    import random
    result = solution.copy()
    i = random.randint(0, 3)
    j = random.randint(0, 3)
    while j == i:
        j = random.randint(0, 3)
    result.operations[i], result.operations[j] = result.operations[j], result.operations[i]
    result.connectivity[i], result.connectivity[j] = result.connectivity[j], result.connectivity[i]
    return result
''',

    "train_loss": '''
def train_loss(solution, ctx):
    """Change an operation to neighbor (+/-1 in op list)."""
    import random
    result = solution.copy()
    idx = random.randint(0, 3)
    current = result.operations[idx]
    if random.random() < 0.5:
        result.operations[idx] = (current + 1) % 9
    else:
        result.operations[idx] = (current - 1) % 9
    return result
''',

    # ==========================================================================
    # REGULARIZATION — simplify/prune
    # ==========================================================================
    "reg_dropout": '''
def reg_dropout(solution, ctx):
    """Set a GNN op to identity (prune a layer)."""
    import random
    result = solution.copy()
    gnn_nodes = [i for i, o in enumerate(result.operations) if o <= 6]
    if gnn_nodes:
        idx = random.choice(gnn_nodes)
        result.operations[idx] = 7  # identity
    return result
''',

    "reg_normalization": '''
def reg_normalization(solution, ctx):
    """Replace complex GNN (arma/k-gnn/cheb) with GCN or SAGE."""
    import random
    result = solution.copy()
    complex_nodes = [i for i, o in enumerate(result.operations) if o in (4, 5, 6)]
    if complex_nodes:
        idx = random.choice(complex_nodes)
        result.operations[idx] = random.choice([0, 2])  # gcn or sage
    return result
''',

    "reg_weight_decay": '''
def reg_weight_decay(solution, ctx):
    """Replace FC with identity (reduce non-graph parameters)."""
    import random
    result = solution.copy()
    fc_nodes = [i for i, o in enumerate(result.operations) if o == 8]
    if fc_nodes:
        idx = random.choice(fc_nodes)
        result.operations[idx] = 7  # identity
    return result
''',

    "reg_structural": '''
def reg_structural(solution, ctx):
    """Enforce minimum 2 GNN operations (guarantee graph awareness)."""
    import random
    result = solution.copy()
    gnn_count = sum(1 for o in result.operations if o <= 6)
    while gnn_count < 2:
        non_gnn = [i for i, o in enumerate(result.operations) if o > 6]
        if not non_gnn:
            break
        idx = random.choice(non_gnn)
        result.operations[idx] = random.choice([0, 1, 3])  # gcn, gat, gin
        gnn_count += 1
    return result
''',

    # ==========================================================================
    # EVALUATION — validate/evaluate
    # ==========================================================================
    "eval_proxy": '''
def eval_proxy(solution, ctx):
    """Validate architecture: clamp connectivity [0,3] and ops [0,8]."""
    result = solution.copy()
    for i in range(4):
        result.connectivity[i] = max(0, min(3, result.connectivity[i]))
        result.operations[i] = max(0, min(8, result.operations[i]))
    return result
''',

    "eval_full": '''
def eval_full(solution, ctx):
    """Trigger full NAS-Bench-Graph evaluation via context."""
    result = solution.copy()
    if hasattr(ctx, 'evaluate'):
        _ = ctx.evaluate(result)
    return result
''',
}


def get_nas_graph_base_operator(role: str) -> str:
    """Get the base operator code for a NAS-Bench-Graph role.

    Args:
        role: NAS role name (e.g., "topo_feedforward")

    Returns:
        Python code string for the base operator.

    Raises:
        KeyError: If role is not found.
    """
    if role not in NAS_GRAPH_BASE_OPERATORS:
        raise KeyError(
            f"Unknown NAS-Graph role: {role}. Available: {ALL_NAS_GRAPH_ROLES}"
        )
    return NAS_GRAPH_BASE_OPERATORS[role].strip()
