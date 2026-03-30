"""Generic operators for architecture-based problems (NAS).

These operators work on NeuralArchitecture objects, modifying their
structure, activations, training config, and regularization settings.

Total: 19 operators covering 18 abstract roles (5 categories).
- Topology: 4 operators
- Activation: 4 operators
- Training: 5 operators
- Regularization: 4 operators
- Evaluation: 2 operators
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ..representations import (
    GenericOperator,
    RepresentationOperators,
    RepresentationType,
)

if TYPE_CHECKING:
    from src.domains.nas.architecture import NeuralArchitecture
    from src.domains.nas.context import NASContext


# =============================================================================
# TOPOLOGY OPERATORS (4) — topo_feedforward, topo_residual, topo_recursive, topo_cell_based
# =============================================================================


def add_layer(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Add a new layer at a random position.

    Inserts a layer with random config from the search space.
    Corresponds to topo_feedforward role.
    """
    from src.domains.nas.architecture import ArchitectureLayer

    result = arch.copy()
    pos = random.randint(0, len(result.layers))
    new_layer = ArchitectureLayer(
        layer_id=len(result.layers),
        layer_type=random.choice(["linear", "conv2d"]),
        units=random.choice([32, 64, 128, 256]),
        activation=random.choice(["relu", "gelu", "silu"]),
        dropout=random.choice([0.0, 0.1, 0.2]),
        normalization=random.choice(["none", "batch", "layer"]),
    )
    result.layers.insert(pos, new_layer)
    # Renumber layer IDs
    for i, layer in enumerate(result.layers):
        layer.layer_id = i
    return result


def remove_layer(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Remove a random layer (keeping at least 2).

    Corresponds to topo_feedforward role (simplification).
    """
    result = arch.copy()
    if len(result.layers) <= 2:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    result.layers.pop(idx)
    # Renumber and fix skip connections
    for i, layer in enumerate(result.layers):
        layer.layer_id = i
    result.skip_connections = [
        (s, t) for s, t in result.skip_connections
        if s < len(result.layers) and t < len(result.layers)
    ]
    return result


def add_skip_connection(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Add a residual/skip connection between two layers.

    Corresponds to topo_residual role.
    """
    result = arch.copy()
    n = len(result.layers)
    if n < 3:
        return result

    # Try to find a valid skip that doesn't already exist
    for _ in range(10):
        src = random.randint(0, n - 3)
        tgt = random.randint(src + 2, n - 1)
        if (src, tgt) not in result.skip_connections:
            result.skip_connections.append((src, tgt))
            return result
    return result


def change_layer_type(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the type of a random layer (linear <-> conv2d).

    Corresponds to topo_cell_based role.
    """
    result = arch.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    layer = result.layers[idx]
    types = ["linear", "conv2d"]
    layer.layer_type = random.choice([t for t in types if t != layer.layer_type] or types)
    return result


# =============================================================================
# ACTIVATION OPERATORS (4) — act_standard, act_modern, act_parametric, act_mixed
# =============================================================================


def change_activation(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the activation of a random layer to a standard one.

    Corresponds to act_standard role.
    """
    result = arch.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    standard = ["relu", "sigmoid", "tanh"]
    result.layers[idx].activation = random.choice(standard)
    return result


def make_activations_uniform(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Set all layers to the same modern activation.

    Corresponds to act_modern role.
    """
    result = arch.copy()
    modern = ["gelu", "silu", "mish"]
    chosen = random.choice(modern)
    for layer in result.layers:
        layer.activation = chosen
    return result


def change_activation_parametric(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change activations to parametric ones (PReLU-like).

    Corresponds to act_parametric role. Uses closest available activations
    since the search space may not include all parametric types.
    """
    result = arch.copy()
    if not result.layers:
        return result
    # Parametric-like: choose from the richer activations
    parametric_like = ["relu", "gelu", "silu", "mish"]
    idx = random.randint(0, len(result.layers) - 1)
    result.layers[idx].activation = random.choice(parametric_like)
    return result


def change_activation_mixed(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Set different activations per layer (mixed search).

    Corresponds to act_mixed role.
    """
    result = arch.copy()
    all_acts = ["relu", "gelu", "silu", "tanh", "sigmoid", "mish"]
    for layer in result.layers:
        layer.activation = random.choice(all_acts)
    return result


# =============================================================================
# TRAINING OPERATORS (5) — train_optimizer, train_schedule, train_augmentation, train_loss
# =============================================================================


def change_optimizer(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the optimizer.

    Corresponds to train_optimizer role.
    """
    result = arch.copy()
    optimizers = ["sgd", "adam", "adamw"]
    result.optimizer = random.choice([o for o in optimizers if o != result.optimizer] or optimizers)
    return result


def adjust_learning_rate(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Adjust learning rate (multiply or divide by 3).

    Corresponds to train_schedule role.
    """
    result = arch.copy()
    lr_choices = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    result.learning_rate = random.choice(lr_choices)
    return result


def change_lr_schedule(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the learning rate schedule.

    Corresponds to train_schedule role.
    """
    result = arch.copy()
    schedules = ["cosine", "step", "warmup_cosine", "cyclical"]
    result.lr_schedule = random.choice(
        [s for s in schedules if s != result.lr_schedule] or schedules
    )
    return result


def change_augmentation(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the data augmentation strategy.

    Corresponds to train_augmentation role.
    """
    result = arch.copy()
    augmentations = ["none", "standard", "cutout", "mixup"]
    result.augmentation = random.choice(
        [a for a in augmentations if a != result.augmentation] or augmentations
    )
    return result


def change_loss_function(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change the loss function.

    Corresponds to train_loss role.
    """
    result = arch.copy()
    losses = ["cross_entropy", "label_smoothing", "focal"]
    result.loss_fn = random.choice(
        [l for l in losses if l != result.loss_fn] or losses
    )
    return result


# =============================================================================
# REGULARIZATION OPERATORS (4) — reg_dropout, reg_normalization, reg_weight_decay, reg_structural
# =============================================================================


def adjust_dropout(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Adjust dropout rate of a random layer.

    Corresponds to reg_dropout role.
    """
    result = arch.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    dropout_choices = [0.0, 0.1, 0.2, 0.3, 0.5]
    result.layers[idx].dropout = random.choice(dropout_choices)
    return result


def change_normalization(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Change normalization of a random layer.

    Corresponds to reg_normalization role.
    """
    result = arch.copy()
    if not result.layers:
        return result
    idx = random.randint(0, len(result.layers) - 1)
    norms = ["none", "batch", "layer", "group"]
    result.layers[idx].normalization = random.choice(norms)
    return result


def adjust_weight_decay(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Adjust weight decay.

    Corresponds to reg_weight_decay role.
    """
    result = arch.copy()
    wd_choices = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    result.weight_decay = random.choice(wd_choices)
    return result


def enforce_structural_constraints(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Enforce structural constraints (max params, simplify if too large).

    Corresponds to reg_structural role.
    """
    result = arch.copy()

    # If too many params, shrink layers
    max_params = 10_000_000
    while result.total_params() > max_params and len(result.layers) > 2:
        # Remove the layer with the most params
        worst_idx = max(range(len(result.layers)), key=lambda i: result.layers[i].param_count())
        result.layers.pop(worst_idx)
        # Renumber
        for i, layer in enumerate(result.layers):
            layer.layer_id = i
        # Fix skip connections
        result.skip_connections = [
            (s, t) for s, t in result.skip_connections
            if s < len(result.layers) and t < len(result.layers)
        ]

    # If still too large, reduce unit sizes
    if result.total_params() > max_params:
        for layer in result.layers:
            if layer.units > 64:
                layer.units = layer.units // 2

    return result


# =============================================================================
# EVALUATION OPERATORS (2) — eval_proxy, eval_full
# =============================================================================


def validate_architecture(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Validate and clean up the architecture.

    Corresponds to eval_proxy role. Ensures architecture is valid
    and fixes any issues (dangling skip connections, etc.).
    """
    result = arch.copy()

    # Fix skip connections
    n = len(result.layers)
    result.skip_connections = [
        (s, t) for s, t in result.skip_connections
        if 0 <= s < n and 0 <= t < n and s < t
    ]

    # Renumber layer IDs
    for i, layer in enumerate(result.layers):
        layer.layer_id = i

    # Clamp dropout
    for layer in result.layers:
        layer.dropout = max(0.0, min(0.5, layer.dropout))

    return result


def evaluate_architecture(arch: "NeuralArchitecture", ctx: "NASContext") -> "NeuralArchitecture":
    """Evaluate architecture (trigger evaluation through context).

    Corresponds to eval_full role. The evaluation itself happens
    through ctx.evaluate(); this operator just ensures the architecture
    is in evaluable form and returns it unchanged.
    """
    result = arch.copy()
    # Trigger evaluation (result is cached in context)
    _ = ctx.evaluate(result)
    return result


# =============================================================================
# REGISTER ALL OPERATORS
# =============================================================================


def _create_architecture_operators() -> RepresentationOperators:
    """Create the collection of all architecture (NAS) operators."""
    ops = RepresentationOperators(representation=RepresentationType.ARCHITECTURE_DAG)

    # Topology (4)
    ops.add_operator(GenericOperator(
        operator_id="add_layer",
        function=add_layer,
        role="topo_feedforward",
        weight=1.0,
        description="Add a new layer at random position",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="remove_layer",
        function=remove_layer,
        role="topo_feedforward",
        weight=1.0,
        description="Remove a random layer",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="add_skip_connection",
        function=add_skip_connection,
        role="topo_residual",
        weight=1.0,
        description="Add a residual/skip connection",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_layer_type",
        function=change_layer_type,
        role="topo_cell_based",
        weight=1.0,
        description="Change layer type (linear <-> conv2d)",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))

    # Activation (4)
    ops.add_operator(GenericOperator(
        operator_id="change_activation",
        function=change_activation,
        role="act_standard",
        weight=1.0,
        description="Change to standard activation",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="make_activations_uniform",
        function=make_activations_uniform,
        role="act_modern",
        weight=1.0,
        description="Set all layers to same modern activation",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_activation_parametric",
        function=change_activation_parametric,
        role="act_parametric",
        weight=1.0,
        description="Change to parametric activation",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_activation_mixed",
        function=change_activation_mixed,
        role="act_mixed",
        weight=1.0,
        description="Set different activation per layer",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))

    # Training (5)
    ops.add_operator(GenericOperator(
        operator_id="change_optimizer",
        function=change_optimizer,
        role="train_optimizer",
        weight=1.0,
        description="Change optimizer (SGD/Adam/AdamW)",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="adjust_learning_rate",
        function=adjust_learning_rate,
        role="train_schedule",
        weight=1.0,
        description="Adjust learning rate",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_lr_schedule",
        function=change_lr_schedule,
        role="train_schedule",
        weight=1.0,
        description="Change LR schedule",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_augmentation",
        function=change_augmentation,
        role="train_augmentation",
        weight=1.0,
        description="Change data augmentation strategy",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_loss_function",
        function=change_loss_function,
        role="train_loss",
        weight=1.0,
        description="Change loss function",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))

    # Regularization (4)
    ops.add_operator(GenericOperator(
        operator_id="adjust_dropout",
        function=adjust_dropout,
        role="reg_dropout",
        weight=1.0,
        description="Adjust dropout rate",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="change_normalization",
        function=change_normalization,
        role="reg_normalization",
        weight=1.0,
        description="Change normalization type",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="adjust_weight_decay",
        function=adjust_weight_decay,
        role="reg_weight_decay",
        weight=1.0,
        description="Adjust weight decay",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="enforce_structural_constraints",
        function=enforce_structural_constraints,
        role="reg_structural",
        weight=1.0,
        description="Enforce parameter/FLOP constraints",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))

    # Evaluation (2)
    ops.add_operator(GenericOperator(
        operator_id="validate_architecture",
        function=validate_architecture,
        role="eval_proxy",
        weight=1.0,
        description="Validate and clean architecture",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))
    ops.add_operator(GenericOperator(
        operator_id="evaluate_architecture",
        function=evaluate_architecture,
        role="eval_full",
        weight=1.0,
        description="Full architecture evaluation",
        representation=RepresentationType.ARCHITECTURE_DAG,
    ))

    return ops


# Singleton instance
ARCHITECTURE_OPERATORS = _create_architecture_operators()
