"""Transfer Learning Module for GEAKG.

Provides adapters for transferring operators between domains,
and extraction of structural knowledge for symbolic transfer.

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/
    Transfer: Domain Adapters - src/geakg/transfer/  <-- YOU ARE HERE

Example:
    from src.geakg.transfer import TransferManager, extract_symbolic_rules

    # Transfer TSP operators to VRP
    manager = TransferManager()
    result = manager.transfer_from_geakg(
        source_snapshot="tsp_snapshot.json",
        target_domain="vrp",
    )

    # Extract symbolic rules for search guidance
    rule_engine = extract_symbolic_rules(snapshot)
"""

from .adapter import DomainAdapter, AdapterConfig, AdaptedOperator
from .l1_knowledge import (
    L1Knowledge,
    L1RoleKnowledge,
    RoleCategory,
    extract_l1_knowledge,
    extract_l1_role_knowledge,
)
from .symbolic_rules import (
    SymbolicRuleEngine,
    SearchState,
    SearchPhase,
    IntensificationLevel,
    extract_symbolic_rules,
    extract_success_frequency,
)
from .symbolic_executor import (
    SymbolicExecutor,
    ExecutionResult,
)
from .snapshot_utils import (
    find_latest_snapshot_with_operators,
    get_synth_operators_from_snapshot,
    get_pheromones_from_snapshot,
)
from .adapters import VRPAdapter, JSSPAdapter
from .transfer_manager import TransferManager

__all__ = [
    # Adapter
    "DomainAdapter",
    "AdapterConfig",
    "AdaptedOperator",
    # L1 Knowledge
    "L1Knowledge",
    "L1RoleKnowledge",
    "RoleCategory",
    "extract_l1_knowledge",
    "extract_l1_role_knowledge",
    # Symbolic Rules & Executor
    "SymbolicRuleEngine",
    "SearchState",
    "SearchPhase",
    "IntensificationLevel",
    "extract_symbolic_rules",
    "extract_success_frequency",
    "SymbolicExecutor",
    "ExecutionResult",
    # Snapshot utilities
    "find_latest_snapshot_with_operators",
    "get_synth_operators_from_snapshot",
    "get_pheromones_from_snapshot",
    # Domain-specific
    "VRPAdapter",
    "JSSPAdapter",
    "TransferManager",
]
