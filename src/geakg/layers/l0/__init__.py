"""L0: MetaGraph Topology Layer for NS-SE.

This layer defines the STRUCTURE of the knowledge graph:
- Abstract roles (11 semantic behaviors)
- MetaGraph topology (how roles connect)
- Initial weights (heuristic preferences from LLM)
- Conditions (adaptive control policies defined by LLM)

L0 = Roles + Transitions + Initial Weights + Conditions

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/  <-- YOU ARE HERE
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/

Usage:
    from src.geakg.layers.l0 import (
        AbstractRole, RoleCategory, ROLE_CATALOG,
        MetaGraph, MetaEdge, InstantiatedGraph,
        EdgeCondition, ExecutionContext, ConditionType,
        create_ils_meta_graph, create_vns_meta_graph,
    )
"""

# === Roles (11 abstract behaviors) ===
from .roles import (
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
from .metagraph import (
    MetaGraph,
    MetaEdge,
    InstantiatedGraph,
)

# === Conditions (adaptive control) ===
from .conditions import (
    ConditionType,
    ComparisonOp,
    EdgeCondition,
    ExecutionContext,
    parse_condition_from_dict,
)

# === Factory functions (ILS, VNS patterns) ===
from .patterns import (
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
    # Patterns
    "create_ils_meta_graph",
    "create_vns_meta_graph",
    "create_hybrid_meta_graph",
]
