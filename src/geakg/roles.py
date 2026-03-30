"""Abstract Roles for Domain-Agnostic Algorithm Composition.

BACKWARD COMPATIBILITY STUB: This module re-exports from src.geakg.layers.l0.roles.
New code should import directly from src.geakg.layers.l0:

    from src.geakg.layers.l0 import AbstractRole, RoleCategory, ROLE_CATALOG

This stub exists for backward compatibility with existing imports.
"""

from src.geakg.layers.l0.roles import (
    AbstractRole,
    RoleCategory,
    RoleNode,
    ROLE_CATALOG,
    VALID_CATEGORY_TRANSITIONS,
    get_role_node,
    get_all_role_nodes,
    get_roles_by_category,
    get_construction_roles,
    get_local_search_roles,
    get_perturbation_roles,
    get_role_description_for_llm,
    is_valid_role_transition,
    get_valid_next_roles,
)

__all__ = [
    "AbstractRole",
    "RoleCategory",
    "RoleNode",
    "ROLE_CATALOG",
    "VALID_CATEGORY_TRANSITIONS",
    "get_role_node",
    "get_all_role_nodes",
    "get_roles_by_category",
    "get_construction_roles",
    "get_local_search_roles",
    "get_perturbation_roles",
    "get_role_description_for_llm",
    "is_valid_role_transition",
    "get_valid_next_roles",
]
