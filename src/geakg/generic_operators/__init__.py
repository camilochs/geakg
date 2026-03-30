"""Generic operators organized by representation type.

This package provides representation-aware but domain-agnostic operators.
Each module corresponds to a RepresentationType and contains operators
that work on any problem with that representation.

Example:
    # Get all permutation operators
    from akg.generic_operators import PERMUTATION_OPERATORS

    # Register for a new domain
    for op in PERMUTATION_OPERATORS.get_all_operators():
        bindings.add_operator(op)
"""

from .permutation import PERMUTATION_OPERATORS
from ..representations import RepresentationOperators, RepresentationType

# Registry of all generic operators by representation
GENERIC_OPERATOR_REGISTRY: dict[RepresentationType, RepresentationOperators] = {
    RepresentationType.PERMUTATION: PERMUTATION_OPERATORS,
    # Future: BINARY_VECTOR, PARTITION, etc.
}


def get_operators_for_representation(
    representation: RepresentationType,
) -> RepresentationOperators:
    """Get all generic operators for a representation type.

    Args:
        representation: The representation type.

    Returns:
        RepresentationOperators containing all operators for that type.

    Raises:
        KeyError: If representation type is not supported.
    """
    if representation not in GENERIC_OPERATOR_REGISTRY:
        raise KeyError(
            f"No generic operators defined for representation: {representation}"
        )
    return GENERIC_OPERATOR_REGISTRY[representation]


__all__ = [
    "PERMUTATION_OPERATORS",
    "GENERIC_OPERATOR_REGISTRY",
    "get_operators_for_representation",
]
