"""Generic operators for NAS-Bench-201 cell-based architectures.

These operators work on CellArchitecture objects (6 edges, 5 operations).
Each operator modifies the cell encoding according to its role semantics.

Total: 18 operators covering 18 abstract roles (5 categories).
- Topology (4): modify edges to change cell structure
- Activation (4): modify edges with semantic bias
- Training (4): incremental modifications
- Regularization (4): simplify/prune operations
- Evaluation (2): validate and evaluate

All operators have signature: (solution: CellArchitecture, ctx) -> CellArchitecture
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
    from src.domains.nas.cell_architecture import CellArchitecture


# Operation indices (matching OPERATIONS in cell_architecture.py)
NONE = 0
SKIP_CONNECT = 1
NOR_CONV_1X1 = 2
NOR_CONV_3X3 = 3
AVG_POOL_3X3 = 4
NUM_OPS = 5
NUM_EDGES = 6


# =============================================================================
# TOPOLOGY OPERATORS (4) — modify cell edge structure
# =============================================================================


def topo_feedforward(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Set a random edge to nor_conv_3x3 (deep convolution).

    Adds a convolutional connection, creating a feedforward path.
    """
    result = arch.copy()
    idx = random.randint(0, NUM_EDGES - 1)
    result.edges[idx] = NOR_CONV_3X3
    return result


def topo_residual(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Set a random edge to skip_connect (residual connection).

    Adds an identity shortcut, inspired by ResNet-style architectures.
    """
    result = arch.copy()
    idx = random.randint(0, NUM_EDGES - 1)
    result.edges[idx] = SKIP_CONNECT
    return result


def topo_recursive(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Swap two random edges (reorganize the cell).

    Restructures information flow without adding/removing operations.
    """
    result = arch.copy()
    i = random.randint(0, NUM_EDGES - 1)
    j = random.randint(0, NUM_EDGES - 1)
    while j == i:
        j = random.randint(0, NUM_EDGES - 1)
    result.edges[i], result.edges[j] = result.edges[j], result.edges[i]
    return result


def topo_cell_based(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Randomize all 6 edges (complete cell redesign).

    Full exploration — generates an entirely new cell topology.
    """
    from src.domains.nas.cell_architecture import CellArchitecture as CA
    return CA.random()


# =============================================================================
# ACTIVATION OPERATORS (4) — modify edges with semantic bias
# =============================================================================


def act_standard(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Replace a 'none' edge with avg_pool_3x3 (standard activation).

    Activates a disconnected edge with a simple pooling operation.
    """
    result = arch.copy()
    none_edges = [i for i, e in enumerate(result.edges) if e == NONE]
    if none_edges:
        idx = random.choice(none_edges)
        result.edges[idx] = AVG_POOL_3X3
    else:
        # If no none edges, change a random edge to avg_pool
        idx = random.randint(0, NUM_EDGES - 1)
        result.edges[idx] = AVG_POOL_3X3
    return result


def act_modern(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Change a convolution edge to nor_conv_1x1 (lightweight operation).

    Replaces heavy 3x3 convs with efficient 1x1 convs.
    """
    result = arch.copy()
    conv_edges = [i for i, e in enumerate(result.edges) if e in (NOR_CONV_3X3,)]
    if conv_edges:
        idx = random.choice(conv_edges)
        result.edges[idx] = NOR_CONV_1X1
    else:
        idx = random.randint(0, NUM_EDGES - 1)
        result.edges[idx] = NOR_CONV_1X1
    return result


def act_parametric(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Replace a skip_connect with nor_conv_3x3 (add learnable parameters).

    Converts identity paths into parametric (learnable) operations.
    """
    result = arch.copy()
    skip_edges = [i for i, e in enumerate(result.edges) if e == SKIP_CONNECT]
    if skip_edges:
        idx = random.choice(skip_edges)
        result.edges[idx] = NOR_CONV_3X3
    else:
        idx = random.randint(0, NUM_EDGES - 1)
        result.edges[idx] = NOR_CONV_3X3
    return result


def act_mixed(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Randomize 2 random edges (mixed exploration).

    Introduces diversity in the activation choices.
    """
    result = arch.copy()
    indices = random.sample(range(NUM_EDGES), min(2, NUM_EDGES))
    for idx in indices:
        result.edges[idx] = random.randint(0, NUM_OPS - 1)
    return result


# =============================================================================
# TRAINING OPERATORS (4) — incremental modifications
# =============================================================================


def train_optimizer(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Replace a 'none' edge with a convolution (more trainable operations).

    Adds computational capacity by activating disconnected edges.
    """
    result = arch.copy()
    none_edges = [i for i, e in enumerate(result.edges) if e == NONE]
    if none_edges:
        idx = random.choice(none_edges)
        result.edges[idx] = random.choice([NOR_CONV_1X1, NOR_CONV_3X3])
    else:
        # If all edges active, upgrade a pool to conv
        pool_edges = [i for i, e in enumerate(result.edges) if e == AVG_POOL_3X3]
        if pool_edges:
            idx = random.choice(pool_edges)
            result.edges[idx] = NOR_CONV_3X3
    return result


def train_schedule(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Swap nor_conv_3x3 <-> nor_conv_1x1 (adjust capacity).

    Trades off between model capacity (3x3) and efficiency (1x1).
    """
    result = arch.copy()
    conv3_edges = [i for i, e in enumerate(result.edges) if e == NOR_CONV_3X3]
    conv1_edges = [i for i, e in enumerate(result.edges) if e == NOR_CONV_1X1]
    if conv3_edges and random.random() < 0.5:
        idx = random.choice(conv3_edges)
        result.edges[idx] = NOR_CONV_1X1
    elif conv1_edges:
        idx = random.choice(conv1_edges)
        result.edges[idx] = NOR_CONV_3X3
    return result


def train_augmentation(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Swap 2 random edges (minor perturbation).

    Analog to data augmentation — small structural change for diversity.
    """
    result = arch.copy()
    i = random.randint(0, NUM_EDGES - 1)
    j = random.randint(0, NUM_EDGES - 1)
    while j == i:
        j = random.randint(0, NUM_EDGES - 1)
    result.edges[i], result.edges[j] = result.edges[j], result.edges[i]
    return result


def train_loss(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Change an edge to its neighbor operation (+/- 1 in operation list).

    Incremental change — minimal perturbation in the operation space.
    """
    result = arch.copy()
    idx = random.randint(0, NUM_EDGES - 1)
    current = result.edges[idx]
    # Move to adjacent operation (wrapping)
    if random.random() < 0.5:
        result.edges[idx] = (current + 1) % NUM_OPS
    else:
        result.edges[idx] = (current - 1) % NUM_OPS
    return result


# =============================================================================
# REGULARIZATION OPERATORS (4) — simplify/prune
# =============================================================================


def reg_dropout(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Set a non-none edge to 'none' (drop path / prune connection).

    Removes a connection, reducing model complexity.
    """
    result = arch.copy()
    non_none = [i for i, e in enumerate(result.edges) if e != NONE]
    if non_none:
        idx = random.choice(non_none)
        result.edges[idx] = NONE
    return result


def reg_normalization(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Replace avg_pool with nor_conv_1x1 (normalized operation).

    Replaces pooling with a more structured parametric operation.
    """
    result = arch.copy()
    pool_edges = [i for i, e in enumerate(result.edges) if e == AVG_POOL_3X3]
    if pool_edges:
        idx = random.choice(pool_edges)
        result.edges[idx] = NOR_CONV_1X1
    else:
        # If no pool edges, find any non-conv edge and make it conv_1x1
        non_conv = [i for i, e in enumerate(result.edges) if e not in (NOR_CONV_1X1, NOR_CONV_3X3)]
        if non_conv:
            idx = random.choice(non_conv)
            result.edges[idx] = NOR_CONV_1X1
    return result


def reg_weight_decay(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Replace nor_conv_3x3 with skip_connect (reduce parameters).

    Trades capacity for efficiency — fewer learnable parameters.
    """
    result = arch.copy()
    conv3_edges = [i for i, e in enumerate(result.edges) if e == NOR_CONV_3X3]
    if conv3_edges:
        idx = random.choice(conv3_edges)
        result.edges[idx] = SKIP_CONNECT
    return result


def reg_structural(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Enforce minimum 4 non-none edges (guarantee connectivity).

    If the cell is too sparse, activates edges with convolutions.
    """
    result = arch.copy()
    non_none_count = sum(1 for e in result.edges if e != NONE)

    while non_none_count < 4:
        none_edges = [i for i, e in enumerate(result.edges) if e == NONE]
        if not none_edges:
            break
        idx = random.choice(none_edges)
        result.edges[idx] = random.choice([NOR_CONV_1X1, NOR_CONV_3X3, SKIP_CONNECT])
        non_none_count += 1

    return result


# =============================================================================
# EVALUATION OPERATORS (2) — validate and evaluate
# =============================================================================


def eval_proxy(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Validate cell encoding (clamp edges to valid range).

    Ensures all edges are in [0, 4].
    """
    result = arch.copy()
    for i in range(NUM_EDGES):
        result.edges[i] = max(0, min(NUM_OPS - 1, result.edges[i]))
    return result


def eval_full(arch: "CellArchitecture", ctx: Any) -> "CellArchitecture":
    """Trigger full NAS-Bench-201 lookup evaluation.

    The actual evaluation happens through ctx.evaluate().
    This operator ensures the architecture is in evaluable form.
    """
    result = arch.copy()
    # Trigger evaluation through context
    if hasattr(ctx, "evaluate"):
        _ = ctx.evaluate(result)
    return result


# =============================================================================
# REGISTER ALL OPERATORS
# =============================================================================


def _create_cell_architecture_operators() -> RepresentationOperators:
    """Create the collection of all NAS-Bench-201 cell operators."""
    ops = RepresentationOperators(representation=RepresentationType.ARCHITECTURE_DAG)

    operator_defs = [
        # Topology (4)
        ("topo_feedforward_cell", topo_feedforward, "topo_feedforward",
         "Set edge to nor_conv_3x3 (feedforward convolution)"),
        ("topo_residual_cell", topo_residual, "topo_residual",
         "Set edge to skip_connect (residual path)"),
        ("topo_recursive_cell", topo_recursive, "topo_recursive",
         "Swap two edges (reorganize cell)"),
        ("topo_cell_based_cell", topo_cell_based, "topo_cell_based",
         "Randomize all 6 edges (full redesign)"),
        # Activation (4)
        ("act_standard_cell", act_standard, "act_standard",
         "Replace none edge with avg_pool_3x3"),
        ("act_modern_cell", act_modern, "act_modern",
         "Change conv to nor_conv_1x1 (lightweight)"),
        ("act_parametric_cell", act_parametric, "act_parametric",
         "Replace skip with nor_conv_3x3 (parametric)"),
        ("act_mixed_cell", act_mixed, "act_mixed",
         "Randomize 2 edges (mixed exploration)"),
        # Training (4)
        ("train_optimizer_cell", train_optimizer, "train_optimizer",
         "Replace none with conv (more trainable ops)"),
        ("train_schedule_cell", train_schedule, "train_schedule",
         "Swap conv_3x3 <-> conv_1x1 (adjust capacity)"),
        ("train_augmentation_cell", train_augmentation, "train_augmentation",
         "Swap 2 edges (minor perturbation)"),
        ("train_loss_cell", train_loss, "train_loss",
         "Neighbor operation change (+/-1)"),
        # Regularization (4)
        ("reg_dropout_cell", reg_dropout, "reg_dropout",
         "Set non-none edge to none (drop path)"),
        ("reg_normalization_cell", reg_normalization, "reg_normalization",
         "Replace avg_pool with nor_conv_1x1"),
        ("reg_weight_decay_cell", reg_weight_decay, "reg_weight_decay",
         "Replace conv_3x3 with skip (reduce params)"),
        ("reg_structural_cell", reg_structural, "reg_structural",
         "Enforce minimum 4 non-none edges"),
        # Evaluation (2)
        ("eval_proxy_cell", eval_proxy, "eval_proxy",
         "Validate and clamp cell encoding"),
        ("eval_full_cell", eval_full, "eval_full",
         "Trigger NAS-Bench-201 lookup evaluation"),
    ]

    for op_id, func, role, desc in operator_defs:
        ops.add_operator(GenericOperator(
            operator_id=op_id,
            function=func,
            role=role,
            weight=1.0,
            description=desc,
            representation=RepresentationType.ARCHITECTURE_DAG,
        ))

    return ops


# Singleton instance
CELL_ARCHITECTURE_OPERATORS = _create_cell_architecture_operators()
