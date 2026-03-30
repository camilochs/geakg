"""Base operators (A₀) for NAS roles.

These are simple, validated, functional operators that serve as
the starting point for AFO (Always-From-Original) generation in NAS.

Each operator:
- Compiles and executes without errors
- Uses only the ctx protocol (evaluate, valid, random_solution, copy)
- Takes (solution, ctx) -> solution signature
- Is simple but functional
- Handles edge cases

18 roles × 1 base operator each = 18 operators.
"""

# All 18 NAS roles in the metagraph
ALL_NAS_ROLES = [
    # Topology (4 roles)
    "topo_feedforward",
    "topo_residual",
    "topo_recursive",
    "topo_cell_based",
    # Activation (4 roles)
    "act_standard",
    "act_modern",
    "act_parametric",
    "act_mixed",
    # Training (4 roles)
    "train_optimizer",
    "train_schedule",
    "train_augmentation",
    "train_loss",
    # Regularization (4 roles)
    "reg_dropout",
    "reg_normalization",
    "reg_weight_decay",
    "reg_structural",
    # Evaluation (2 roles)
    "eval_proxy",
    "eval_full",
]

# Base operators by NAS role - DOMAIN-AGNOSTIC using architecture API
#
# These operators modify NeuralArchitecture objects.
# They only use:
#   - solution.copy() -> NeuralArchitecture  (deep copy)
#   - solution.layers -> list[ArchitectureLayer]  (layer access)
#   - solution.skip_connections -> list[tuple]
#   - solution.optimizer, learning_rate, etc. (hyperparameters)
#   - ctx.evaluate(solution) -> float  (fitness)
#   - ctx.valid(solution) -> bool  (validity check)
#
NAS_BASE_OPERATORS = {
    # ==========================================================================
    # TOPOLOGY - modify network structure
    # ==========================================================================
    "topo_feedforward": '''
def topo_feedforward(solution, ctx):
    """Add or remove a layer to modify the feedforward structure."""
    import random
    from src.domains.nas.architecture import ArchitectureLayer
    result = solution.copy()
    if random.random() < 0.5 and len(result.layers) > 2:
        # Remove a random layer
        idx = random.randint(0, len(result.layers) - 1)
        result.layers.pop(idx)
    else:
        # Add a layer at random position
        pos = random.randint(0, len(result.layers))
        new_layer = ArchitectureLayer(
            layer_id=pos,
            layer_type=random.choice(["linear", "conv2d"]),
            units=random.choice([32, 64, 128, 256]),
            activation=random.choice(["relu", "gelu", "silu"]),
        )
        result.layers.insert(pos, new_layer)
    # Renumber layers
    for i, layer in enumerate(result.layers):
        layer.layer_id = i
    # Fix skip connections
    n = len(result.layers)
    result.skip_connections = [(s, t) for s, t in result.skip_connections if s < n and t < n]
    return result
''',

    "topo_residual": '''
def topo_residual(solution, ctx):
    """Add or remove a skip/residual connection."""
    import random
    result = solution.copy()
    n = len(result.layers)
    if n < 3:
        return result
    if result.skip_connections and random.random() < 0.3:
        # Remove a random skip connection
        idx = random.randint(0, len(result.skip_connections) - 1)
        result.skip_connections.pop(idx)
    else:
        # Add a new skip connection
        for _ in range(10):
            src = random.randint(0, n - 3)
            tgt = random.randint(src + 2, n - 1)
            if (src, tgt) not in result.skip_connections:
                result.skip_connections.append((src, tgt))
                break
    return result
''',

    "topo_recursive": '''
def topo_recursive(solution, ctx):
    """Change a layer to a recurrent-like type (simulate via structure change)."""
    import random
    result = solution.copy()
    if not result.layers:
        return result
    # Pick a random layer and change its properties for recurrent-like behavior
    idx = random.randint(0, len(result.layers) - 1)
    layer = result.layers[idx]
    layer.layer_type = random.choice(["linear", "conv2d"])
    layer.units = random.choice([64, 128, 256])
    # Add self-skip if possible (simulates recurrence)
    if idx > 0 and (idx - 1, idx) not in result.skip_connections:
        result.skip_connections.append((idx - 1, idx))
    return result
''',

    "topo_cell_based": '''
def topo_cell_based(solution, ctx):
    """Restructure architecture to replicate a cell pattern."""
    import random
    from src.domains.nas.architecture import ArchitectureLayer
    result = solution.copy()
    if len(result.layers) < 2:
        return result
    # Take the first 2 layers as a "cell" template and replicate
    cell_size = min(2, len(result.layers))
    cell = result.layers[:cell_size]
    n_cells = random.randint(2, 4)
    new_layers = []
    for c in range(n_cells):
        for layer in cell:
            new_layers.append(ArchitectureLayer(
                layer_id=len(new_layers),
                layer_type=layer.layer_type,
                units=layer.units,
                activation=layer.activation,
                dropout=layer.dropout,
                normalization=layer.normalization,
            ))
    result.layers = new_layers
    result.skip_connections = []
    # Add inter-cell skip connections
    for c in range(n_cells - 1):
        src = c * cell_size
        tgt = (c + 1) * cell_size + cell_size - 1
        if tgt < len(result.layers):
            result.skip_connections.append((src, tgt))
    return result
''',

    # ==========================================================================
    # ACTIVATION - modify activation functions
    # ==========================================================================
    "act_standard": '''
def act_standard(solution, ctx):
    """Change a random layer to use a standard activation."""
    import random
    result = solution.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    result.layers[idx].activation = random.choice(["relu", "sigmoid", "tanh"])
    return result
''',

    "act_modern": '''
def act_modern(solution, ctx):
    """Set all layers to a modern activation (GELU, SiLU, Mish)."""
    import random
    result = solution.copy()
    chosen = random.choice(["gelu", "silu", "mish"])
    for layer in result.layers:
        layer.activation = chosen
    return result
''',

    "act_parametric": '''
def act_parametric(solution, ctx):
    """Change a random layer to a richer activation type."""
    import random
    result = solution.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    result.layers[idx].activation = random.choice(["relu", "gelu", "silu", "mish"])
    return result
''',

    "act_mixed": '''
def act_mixed(solution, ctx):
    """Assign different activations per layer."""
    import random
    result = solution.copy()
    all_acts = ["relu", "gelu", "silu", "tanh", "sigmoid", "mish"]
    for layer in result.layers:
        layer.activation = random.choice(all_acts)
    return result
''',

    # ==========================================================================
    # TRAINING - modify training hyperparameters
    # ==========================================================================
    "train_optimizer": '''
def train_optimizer(solution, ctx):
    """Change the optimizer."""
    import random
    result = solution.copy()
    optimizers = ["sgd", "adam", "adamw"]
    result.optimizer = random.choice([o for o in optimizers if o != result.optimizer] or optimizers)
    return result
''',

    "train_schedule": '''
def train_schedule(solution, ctx):
    """Change the learning rate and schedule."""
    import random
    result = solution.copy()
    result.learning_rate = random.choice([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    result.lr_schedule = random.choice(["cosine", "step", "warmup_cosine", "cyclical"])
    return result
''',

    "train_augmentation": '''
def train_augmentation(solution, ctx):
    """Change the data augmentation strategy."""
    import random
    result = solution.copy()
    augmentations = ["none", "standard", "cutout", "mixup"]
    result.augmentation = random.choice(
        [a for a in augmentations if a != result.augmentation] or augmentations
    )
    return result
''',

    "train_loss": '''
def train_loss(solution, ctx):
    """Change the loss function."""
    import random
    result = solution.copy()
    losses = ["cross_entropy", "label_smoothing", "focal"]
    result.loss_fn = random.choice([l for l in losses if l != result.loss_fn] or losses)
    return result
''',

    # ==========================================================================
    # REGULARIZATION - modify regularization settings
    # ==========================================================================
    "reg_dropout": '''
def reg_dropout(solution, ctx):
    """Adjust dropout of a random layer."""
    import random
    result = solution.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    result.layers[idx].dropout = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
    return result
''',

    "reg_normalization": '''
def reg_normalization(solution, ctx):
    """Change normalization of a random layer."""
    import random
    result = solution.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    result.layers[idx].normalization = random.choice(["none", "batch", "layer", "group"])
    return result
''',

    "reg_weight_decay": '''
def reg_weight_decay(solution, ctx):
    """Adjust weight decay."""
    import random
    result = solution.copy()
    result.weight_decay = random.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    return result
''',

    "reg_structural": '''
def reg_structural(solution, ctx):
    """Enforce structural constraints: remove layers if too many params."""
    result = solution.copy()
    max_params = 10_000_000
    while result.total_params() > max_params and len(result.layers) > 2:
        # Remove the heaviest layer
        worst = max(range(len(result.layers)), key=lambda i: result.layers[i].param_count())
        result.layers.pop(worst)
        for i, layer in enumerate(result.layers):
            layer.layer_id = i
        n = len(result.layers)
        result.skip_connections = [(s, t) for s, t in result.skip_connections if s < n and t < n]
    return result
''',

    # ==========================================================================
    # EVALUATION - validate/evaluate architectures
    # ==========================================================================
    "eval_proxy": '''
def eval_proxy(solution, ctx):
    """Validate architecture and clean up issues."""
    result = solution.copy()
    n = len(result.layers)
    # Fix skip connections
    result.skip_connections = [
        (s, t) for s, t in result.skip_connections
        if 0 <= s < n and 0 <= t < n and s < t
    ]
    # Renumber layer IDs
    for i, layer in enumerate(result.layers):
        layer.layer_id = i
    # Clamp dropout values
    for layer in result.layers:
        layer.dropout = max(0.0, min(0.5, layer.dropout))
    return result
''',

    "eval_full": '''
def eval_full(solution, ctx):
    """Trigger full evaluation (through context)."""
    result = solution.copy()
    # The evaluation happens via ctx.evaluate(); we just return the arch.
    _ = ctx.evaluate(result)
    return result
''',
}


def get_nas_base_operator(role: str) -> str:
    """Get the base operator code for a NAS role.

    Args:
        role: NAS role name (e.g., "topo_feedforward")

    Returns:
        Python code string for the base operator.

    Raises:
        KeyError: If role is not found.
    """
    if role not in NAS_BASE_OPERATORS:
        raise KeyError(f"Unknown NAS role: {role}. Available: {ALL_NAS_ROLES}")
    return NAS_BASE_OPERATORS[role].strip()
