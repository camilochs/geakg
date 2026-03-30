"""NAS operator dispatcher.

Routes operator names to their implementation in generic_operators/architecture.py.
Called by execution.py's _apply_architecture_operator().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.domains.nas.architecture import NeuralArchitecture
    from src.domains.nas.context import NASContext

# Lazy import to avoid circular dependencies
_OPERATOR_MAP: dict[str, Any] | None = None


def _get_operator_map() -> dict:
    """Build operator dispatch table (lazy, cached)."""
    global _OPERATOR_MAP
    if _OPERATOR_MAP is not None:
        return _OPERATOR_MAP

    from src.geakg.generic_operators.architecture import (
        # Topology
        add_layer,
        remove_layer,
        add_skip_connection,
        change_layer_type,
        # Activation
        change_activation,
        make_activations_uniform,
        change_activation_parametric,
        change_activation_mixed,
        # Training
        change_optimizer,
        adjust_learning_rate,
        change_lr_schedule,
        change_augmentation,
        change_loss_function,
        # Regularization
        adjust_dropout,
        change_normalization,
        adjust_weight_decay,
        enforce_structural_constraints,
        # Evaluation
        validate_architecture,
        evaluate_architecture,
    )

    _OPERATOR_MAP = {
        # Topology operators
        "add_layer": add_layer,
        "remove_layer": remove_layer,
        "add_skip_connection": add_skip_connection,
        "change_layer_type": change_layer_type,
        # Activation operators
        "change_activation": change_activation,
        "make_activations_uniform": make_activations_uniform,
        "change_activation_parametric": change_activation_parametric,
        "change_activation_mixed": change_activation_mixed,
        # Training operators
        "change_optimizer": change_optimizer,
        "adjust_learning_rate": adjust_learning_rate,
        "change_lr_schedule": change_lr_schedule,
        "change_augmentation": change_augmentation,
        "change_loss_function": change_loss_function,
        # Regularization operators
        "adjust_dropout": adjust_dropout,
        "change_normalization": change_normalization,
        "adjust_weight_decay": adjust_weight_decay,
        "enforce_structural_constraints": enforce_structural_constraints,
        # Evaluation operators
        "validate_architecture": validate_architecture,
        "evaluate_architecture": evaluate_architecture,
    }
    return _OPERATOR_MAP


def apply_nas_operator(
    op: str,
    solution: "NeuralArchitecture",
    ctx: "NASContext",
) -> "NeuralArchitecture | None":
    """Apply a NAS operator to an architecture.

    Args:
        op: Operator name.
        solution: Current neural architecture.
        ctx: NASContext with evaluate, valid, random_solution methods.

    Returns:
        Modified architecture if successful, None if operator unknown.
    """
    op_map = _get_operator_map()

    func = op_map.get(op)
    if func is None:
        logger.debug(f"[NAS-OPS] Unknown operator: {op}")
        return None

    try:
        result = func(solution, ctx)
        if result is not None and ctx.valid(result):
            return result
        logger.debug(f"[NAS-OPS] Operator {op} produced invalid architecture")
        return solution  # Return original if invalid
    except Exception as e:
        logger.debug(f"[NAS-OPS] Operator {op} failed: {e}")
        return solution
