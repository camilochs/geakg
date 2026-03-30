"""Minimal A₀ base operators for NAS-Bench-Graph GNN architectures.

These operators have NO domain knowledge about GNN operations or graph
structure.  Each operator applies a distinct *mutation bias* — a different
way to perturb an architecture — so that ACO can learn which bias is
useful at each stage of the search.  The LLM-generated L1 operators add
the actual algorithmic intelligence (GNN-aware heuristics, etc.).

Evaluation operators are identity (noop) — evaluation is handled by the
ACO loop, not by operators.

Operations: {0..8}   (9 values)
Connectivity: [0,3]  (4 values, 0=input, 1-3=prior computing node)

All operators have signature: (solution: GraphArchitecture, ctx) -> GraphArchitecture
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
    from src.domains.nas.graph_architecture import GraphArchitecture


NUM_OPS = 9
NUM_NODES = 4
MAX_CONN = 4


# =============================================================================
# TOPOLOGY OPERATORS (4) — connectivity mutation with different biases
# =============================================================================


def topo_feedforward(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: set 1 connectivity to input (0)."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    result.connectivity[idx] = 0
    return result


def topo_residual(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: set 1 connectivity to chain pattern (node i connects to i)."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    result.connectivity[idx] = idx  # chain: node i reads from node i
    return result


def topo_recursive(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: swap 2 connectivity values."""
    result = arch.copy()
    i, j = random.sample(range(NUM_NODES), 2)
    result.connectivity[i], result.connectivity[j] = (
        result.connectivity[j], result.connectivity[i],
    )
    return result


def topo_cell_based(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: randomize all connectivity values."""
    result = arch.copy()
    result.connectivity = [random.randint(0, MAX_CONN - 1) for _ in range(NUM_NODES)]
    return result


# =============================================================================
# ACTIVATION OPERATORS (4) — operation mutation with different biases
# =============================================================================


def act_standard(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: flip 1 operation to a random value."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    result.operations[idx] = random.randint(0, NUM_OPS - 1)
    return result


def act_modern(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: nudge 1 operation by ±1 (neighbor mutation)."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    delta = random.choice([-1, 1])
    result.operations[idx] = max(0, min(NUM_OPS - 1, result.operations[idx] + delta))
    return result


def act_parametric(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: swap 2 operation values."""
    result = arch.copy()
    i, j = random.sample(range(NUM_NODES), 2)
    result.operations[i], result.operations[j] = (
        result.operations[j], result.operations[i],
    )
    return result


def act_mixed(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: randomize 2 operation values."""
    result = arch.copy()
    indices = random.sample(range(NUM_NODES), 2)
    for idx in indices:
        result.operations[idx] = random.randint(0, NUM_OPS - 1)
    return result


# =============================================================================
# TRAINING OPERATORS (4) — combined mutation (conn + ops)
# =============================================================================


def train_optimizer(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: flip 1 operation + 1 connectivity (joint mutation)."""
    result = arch.copy()
    oi = random.randint(0, NUM_NODES - 1)
    ci = random.randint(0, NUM_NODES - 1)
    result.operations[oi] = random.randint(0, NUM_OPS - 1)
    result.connectivity[ci] = random.randint(0, MAX_CONN - 1)
    return result


def train_schedule(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: swap a full node (connectivity + operation) between 2 positions."""
    result = arch.copy()
    i, j = random.sample(range(NUM_NODES), 2)
    result.connectivity[i], result.connectivity[j] = (
        result.connectivity[j], result.connectivity[i],
    )
    result.operations[i], result.operations[j] = (
        result.operations[j], result.operations[i],
    )
    return result


def train_augmentation(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: copy a random node's (conn, op) to another position."""
    result = arch.copy()
    src, dst = random.sample(range(NUM_NODES), 2)
    result.connectivity[dst] = result.connectivity[src]
    result.operations[dst] = result.operations[src]
    return result


def train_loss(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: flip 2 operations simultaneously (multi-flip)."""
    result = arch.copy()
    indices = random.sample(range(NUM_NODES), 2)
    for idx in indices:
        result.operations[idx] = random.randint(0, NUM_OPS - 1)
    return result


# =============================================================================
# REGULARIZATION OPERATORS (4) — structural perturbation
# =============================================================================


def reg_dropout(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: reset 1 random node to (conn=0, op=0)."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    result.connectivity[idx] = 0
    result.operations[idx] = 0
    return result


def reg_normalization(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: shuffle all operations (keep connectivity)."""
    result = arch.copy()
    ops = list(result.operations)
    random.shuffle(ops)
    result.operations = ops
    return result


def reg_weight_decay(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: nudge 1 connectivity by ±1 (neighbor mutation)."""
    result = arch.copy()
    idx = random.randint(0, NUM_NODES - 1)
    delta = random.choice([-1, 1])
    result.connectivity[idx] = max(0, min(MAX_CONN - 1, result.connectivity[idx] + delta))
    return result


def reg_structural(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Bias: shuffle all connectivity (keep operations)."""
    result = arch.copy()
    conn = list(result.connectivity)
    random.shuffle(conn)
    result.connectivity = conn
    return result


# =============================================================================
# EVALUATION OPERATORS (2) — noop (identity)
# =============================================================================


def eval_proxy(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Noop: return architecture unchanged."""
    return arch.copy()


def eval_full(arch: "GraphArchitecture", ctx: Any) -> "GraphArchitecture":
    """Noop: return architecture unchanged."""
    return arch.copy()


# =============================================================================
# REGISTER ALL OPERATORS
# =============================================================================


def _create_graph_architecture_operators() -> RepresentationOperators:
    """Create the minimal A₀ operators for NAS-Bench-Graph."""
    ops = RepresentationOperators(representation=RepresentationType.ARCHITECTURE_DAG)

    operator_defs = [
        # Topology (4) — connectivity with different biases
        ("topo_feedforward_graph", topo_feedforward, "topo_feedforward",
         "Bias: set 1 connectivity to input (0)"),
        ("topo_residual_graph", topo_residual, "topo_residual",
         "Bias: set 1 connectivity to chain pattern"),
        ("topo_recursive_graph", topo_recursive, "topo_recursive",
         "Bias: swap 2 connectivity values"),
        ("topo_cell_based_graph", topo_cell_based, "topo_cell_based",
         "Bias: randomize all connectivity values"),
        # Activation (4) — operation with different biases
        ("act_standard_graph", act_standard, "act_standard",
         "Bias: flip 1 operation to random value"),
        ("act_modern_graph", act_modern, "act_modern",
         "Bias: nudge 1 operation by +-1"),
        ("act_parametric_graph", act_parametric, "act_parametric",
         "Bias: swap 2 operation values"),
        ("act_mixed_graph", act_mixed, "act_mixed",
         "Bias: randomize 2 operation values"),
        # Training (4) — combined mutations
        ("train_optimizer_graph", train_optimizer, "train_optimizer",
         "Bias: flip 1 op + 1 conn (joint mutation)"),
        ("train_schedule_graph", train_schedule, "train_schedule",
         "Bias: swap full node between 2 positions"),
        ("train_augmentation_graph", train_augmentation, "train_augmentation",
         "Bias: copy node to another position"),
        ("train_loss_graph", train_loss, "train_loss",
         "Bias: flip 2 operations simultaneously"),
        # Regularization (4) — structural perturbation
        ("reg_dropout_graph", reg_dropout, "reg_dropout",
         "Bias: reset 1 node to (conn=0, op=0)"),
        ("reg_normalization_graph", reg_normalization, "reg_normalization",
         "Bias: shuffle all operations"),
        ("reg_weight_decay_graph", reg_weight_decay, "reg_weight_decay",
         "Bias: nudge 1 connectivity by +-1"),
        ("reg_structural_graph", reg_structural, "reg_structural",
         "Bias: shuffle all connectivity"),
        # Evaluation (2) — noop
        ("eval_proxy_graph", eval_proxy, "eval_proxy",
         "Noop: return architecture unchanged"),
        ("eval_full_graph", eval_full, "eval_full",
         "Noop: return architecture unchanged"),
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
GRAPH_ARCHITECTURE_OPERATORS = _create_graph_architecture_operators()
