"""L1: BACKWARD COMPATIBILITY STUB.

DEPRECATED: This module re-exports from src.geakg.layers.l0.
New code should import directly from src.geakg.layers.l0:

    from src.geakg.layers.l0 import (
        AbstractRole, RoleCategory, ROLE_CATALOG,
        MetaGraph, MetaEdge, InstantiatedGraph,
        EdgeCondition, ExecutionContext, ConditionType,
        create_ils_meta_graph, create_vns_meta_graph,
    )

This stub exists for backward compatibility with existing imports.
"""

# === Roles (11 abstract behaviors) ===
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

# === MetaGraph (topology + weights) ===
from src.geakg.layers.l0.metagraph import (
    MetaGraph,
    MetaEdge,
    InstantiatedGraph,
)

# === Conditions (adaptive control) ===
from src.geakg.layers.l0.conditions import (
    ConditionType,
    ComparisonOp,
    EdgeCondition,
    ExecutionContext,
    parse_condition_from_dict,
)

# === Factory functions (ILS, VNS patterns) ===
from src.geakg.layers.l0.patterns import (
    create_ils_meta_graph,
    create_vns_meta_graph,
    create_hybrid_meta_graph,
)

__all__ = [
    # Roles
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
    # MetaGraph
    "MetaGraph",
    "MetaEdge",
    "InstantiatedGraph",
    # Conditions
    "ConditionType",
    "ComparisonOp",
    "EdgeCondition",
    "ExecutionContext",
    "parse_condition_from_dict",
    # Factory
    "create_ils_meta_graph",
    "create_vns_meta_graph",
    "create_hybrid_meta_graph",
]
