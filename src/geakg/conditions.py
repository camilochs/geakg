"""Conditional Transitions for Algorithmic Knowledge Graphs.

BACKWARD COMPATIBILITY STUB: This module re-exports from src.geakg.layers.l0.conditions.
New code should import directly from src.geakg.layers.l0:

    from src.geakg.layers.l0 import EdgeCondition, ExecutionContext, ConditionType

This stub exists for backward compatibility with existing imports.
"""

from src.geakg.layers.l0.conditions import (
    ConditionType,
    ComparisonOp,
    EdgeCondition,
    ExecutionContext,
    parse_condition_from_dict,
)

__all__ = [
    "ConditionType",
    "ComparisonOp",
    "EdgeCondition",
    "ExecutionContext",
    "parse_condition_from_dict",
]
