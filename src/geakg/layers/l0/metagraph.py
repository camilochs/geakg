"""Meta-Graph: Domain-Agnostic Algorithm Composition Structure.

Part of L1 (Unified Knowledge Layer): This module defines the MetaGraph - a
domain-agnostic representation of algorithm composition that can be instantiated
for any domain through bindings.

Key insight: The LLM generates a MetaGraph using abstract roles. This same
MetaGraph can then be instantiated for TSP, JSSP, or any domain by providing
appropriate bindings. This is explicit transfer learning through abstraction.

Architecture:
    MetaGraph (domain-agnostic)
        |
        | instantiate(domain="tsp")
        v
    InstantiatedGraph (domain-specific)
        |
        | used by MetaACOSelector
        v
    Concrete operator sequences

Generalization: MetaEdge source/target accept both AbstractRole and str.
When a RoleSchema is provided, validation uses the schema; otherwise
falls back to AbstractRole enum for backward compatibility.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union
from pydantic import BaseModel, Field

from .roles import (
    AbstractRole,
    RoleNode,
    get_role_node,
    is_valid_role_transition,
)
from .conditions import EdgeCondition, ConditionType

if TYPE_CHECKING:
    from src.geakg.bindings import DomainBindings, BindingRegistry
    from src.geakg.core.role_schema import RoleSchema


def _role_to_str(role: Union[str, AbstractRole]) -> str:
    """Convert a role (str or AbstractRole) to its string value."""
    if isinstance(role, AbstractRole):
        return role.value
    return role


class MetaEdge(BaseModel):
    """An edge in the MetaGraph connecting two abstract roles.

    source/target accept both AbstractRole and plain str for generality.
    Supports weights and conditions for adaptive control.
    """

    source: Union[str, AbstractRole] = Field(..., description="Source role")
    target: Union[str, AbstractRole] = Field(..., description="Target role")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Transition weight")

    # Conditions for adaptive control
    conditions: list[EdgeCondition] = Field(
        default_factory=list,
        description="Conditions that boost this edge's probability"
    )
    condition_boost: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier when conditions are met"
    )

    # Metadata
    reasoning: str = Field(
        default="",
        description="LLM's reasoning for this transition"
    )

    @property
    def source_str(self) -> str:
        """Source role as string."""
        return _role_to_str(self.source)

    @property
    def target_str(self) -> str:
        """Target role as string."""
        return _role_to_str(self.target)

    def has_conditions(self) -> bool:
        """Check if this edge has any conditions."""
        return len(self.conditions) > 0

    def get_effective_weight(self, context: "ExecutionContext") -> float:
        """Get weight adjusted for conditions.

        Args:
            context: ExecutionContext with runtime metrics

        Returns:
            Weight multiplied by boost if any condition is met.
        """
        if not self.conditions:
            return self.weight

        for cond in self.conditions:
            if cond.evaluate(context):
                return self.weight * self.condition_boost

        return self.weight

    def __repr__(self) -> str:
        cond_str = f" [cond: {len(self.conditions)}]" if self.conditions else ""
        return f"MetaEdge({self.source_str} -> {self.target_str}, w={self.weight:.2f}{cond_str})"


class MetaGraph(BaseModel):
    """Domain-agnostic graph of abstract roles.

    This is what the LLM generates - a composition of abstract
    behaviors, not specific operators. Can be instantiated for any domain.

    When role_schema is provided, all validation and queries use the schema.
    When role_schema is None, falls back to AbstractRole enum (backward compatible).
    """

    nodes: dict[str, RoleNode] = Field(
        default_factory=dict,
        description="Role nodes by role value"
    )
    edges: dict[tuple, MetaEdge] = Field(
        default_factory=dict,
        description="Edges indexed by (source, target) role values"
    )

    # Metadata
    name: str = Field(default="meta_graph", description="Graph name/identifier")
    description: str = Field(default="", description="LLM-generated description")
    llm_reasoning: str = Field(default="", description="LLM's design reasoning")

    model_config = {"arbitrary_types_allowed": True}

    # RoleSchema (set after construction since Pydantic BaseModel)
    _role_schema: Optional["RoleSchema"] = None

    @property
    def role_schema(self) -> Optional["RoleSchema"]:
        return self._role_schema

    @role_schema.setter
    def role_schema(self, schema: Optional["RoleSchema"]) -> None:
        self._role_schema = schema

    def add_role(self, role: Union[str, AbstractRole]) -> None:
        """Add a role node to the graph.

        Accepts both AbstractRole and str. When a RoleSchema is set,
        str roles create a generic RoleNode from schema metadata.
        """
        role_str = _role_to_str(role)
        if role_str in self.nodes:
            return

        # Try AbstractRole first (backward compatible)
        if isinstance(role, AbstractRole):
            self.nodes[role_str] = get_role_node(role)
            return

        try:
            self.nodes[role_str] = get_role_node(AbstractRole(role_str))
            return
        except ValueError:
            pass

        # Use schema for non-optimization roles
        if self._role_schema is not None and self._role_schema.is_valid_role(role_str):
            meta = self._role_schema.get_role_metadata(role_str)
            from .roles import RoleCategory
            # Use a generic category string if not a standard RoleCategory
            try:
                cat = RoleCategory(meta.get("category", "local_search"))
            except ValueError:
                cat = RoleCategory.LOCAL_SEARCH  # Fallback for non-optimization categories

            self.nodes[role_str] = RoleNode(
                role=role_str,  # type: ignore[arg-type] – RoleNode.role accepts str too via schema
                description=meta.get("description", role_str),
                category=cat,
                expected_cost=meta.get("expected_cost", "O(n)"),
                exploration_bias=meta.get("exploration_bias", 0.5),
            )
            return

        raise ValueError(f"Unknown role: {role_str} (not in AbstractRole or RoleSchema)")

    def add_role_generic(self, role_id: str, description: str = "",
                         category: str = "", expected_cost: str = "O(n)",
                         exploration_bias: float = 0.5) -> None:
        """Add a generic role node (not tied to AbstractRole enum).

        Used by NAS and other non-optimization schemas.
        """
        if role_id in self.nodes:
            return
        from .roles import RoleCategory
        try:
            cat = RoleCategory(category)
        except ValueError:
            cat = RoleCategory.LOCAL_SEARCH
        self.nodes[role_id] = RoleNode(
            role=role_id,  # type: ignore[arg-type]
            description=description or role_id,
            category=cat,
            expected_cost=expected_cost,
            exploration_bias=exploration_bias,
        )

    def add_edge(self, edge: MetaEdge) -> None:
        """Add an edge between roles."""
        src_str = edge.source_str
        tgt_str = edge.target_str

        # Ensure nodes exist
        self._ensure_node(edge.source)
        self._ensure_node(edge.target)

        key = (src_str, tgt_str)
        self.edges[key] = edge

    def _ensure_node(self, role: Union[str, AbstractRole]) -> None:
        """Ensure a node exists for this role."""
        role_str = _role_to_str(role)
        if role_str not in self.nodes:
            try:
                self.add_role(role)
            except ValueError:
                # For unknown roles, add minimal node
                self.add_role_generic(role_str)

    def get_edge(self, source: Union[str, AbstractRole],
                 target: Union[str, AbstractRole]) -> Optional[MetaEdge]:
        """Get edge between two roles."""
        key = (_role_to_str(source), _role_to_str(target))
        return self.edges.get(key)

    def get_outgoing_edges(self, role: Union[str, AbstractRole]) -> list[MetaEdge]:
        """Get all edges originating from a role."""
        role_str = _role_to_str(role)
        edges = []
        for (src, _), edge in self.edges.items():
            if src == role_str:
                edges.append(edge)
        return edges

    def get_incoming_edges(self, role: Union[str, AbstractRole]) -> list[MetaEdge]:
        """Get all edges targeting a role."""
        role_str = _role_to_str(role)
        edges = []
        for (_, tgt), edge in self.edges.items():
            if tgt == role_str:
                edges.append(edge)
        return edges

    def get_successors(self, role: Union[str, AbstractRole]) -> list[str]:
        """Get all roles reachable from a role in one step.

        Returns string role IDs (not AbstractRole) for generality.
        """
        role_str = _role_to_str(role)
        successors = []
        for (src, tgt) in self.edges:
            if src == role_str:
                successors.append(tgt)
        return successors

    def get_successors_as_roles(self, role: AbstractRole) -> list[AbstractRole]:
        """Get successors as AbstractRole enums (backward compat)."""
        return [AbstractRole(s) for s in self.get_successors(role)]

    def get_entry_roles(self) -> list[str]:
        """Get entry roles using the schema, or construction roles as fallback."""
        if self._role_schema is not None:
            entry = self._role_schema.get_entry_roles()
            return [r for r in entry if r in self.nodes]

        # Backward compatible: construction roles
        return [
            r for r, node in self.nodes.items()
            if node.is_construction()
        ]

    def get_construction_roles(self) -> list[AbstractRole]:
        """Get all construction roles in this graph (backward compat)."""
        return [
            AbstractRole(r) for r, node in self.nodes.items()
            if node.is_construction()
        ]

    def has_edge(self, source: Union[str, AbstractRole],
                 target: Union[str, AbstractRole]) -> bool:
        """Check if edge exists."""
        return (_role_to_str(source), _role_to_str(target)) in self.edges

    def instantiate(self, domain: str) -> "InstantiatedGraph":
        """Instantiate this meta-graph for a specific domain.

        Args:
            domain: Domain name (e.g., "tsp", "jssp")

        Returns:
            InstantiatedGraph ready for execution
        """
        from src.geakg.bindings import BindingRegistry

        registry = BindingRegistry()
        bindings = registry.get_domain(domain)

        if bindings is None:
            raise ValueError(f"Unknown domain: {domain}")

        return InstantiatedGraph(
            meta_graph=self,
            bindings=bindings,
        )

    def validate_transitions(self) -> list[str]:
        """Check if all transitions follow role rules.

        Uses RoleSchema if available, otherwise AbstractRole-based validation.

        Returns:
            List of warnings for invalid transitions
        """
        warnings = []
        for (src, tgt), edge in self.edges.items():
            if self._role_schema is not None:
                if not self._role_schema.is_valid_transition(src, tgt):
                    warnings.append(
                        f"Unusual transition: {src} -> {tgt} "
                        f"(not valid per schema)"
                    )
            else:
                try:
                    source_role = AbstractRole(src)
                    target_role = AbstractRole(tgt)
                    if not is_valid_role_transition(source_role, target_role):
                        warnings.append(
                            f"Unusual transition: {src} -> {tgt} "
                            f"(may not follow standard optimization patterns)"
                        )
                except ValueError:
                    pass  # Non-optimization roles, can't validate without schema
        return warnings

    def __repr__(self) -> str:
        return f"MetaGraph(name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"MetaGraph: {self.name}",
            f"  Roles: {len(self.nodes)}",
            f"  Edges: {len(self.edges)}",
            "",
            "Edges:",
        ]
        for (src, tgt), edge in sorted(self.edges.items()):
            cond = f" [cond]" if edge.conditions else ""
            lines.append(f"  {src} -> {tgt} (w={edge.weight:.2f}{cond})")

        return "\n".join(lines)


class InstantiatedGraph:
    """A MetaGraph instantiated for a specific domain.

    Combines the abstract structure (MetaGraph) with domain-specific
    operator bindings (DomainBindings).
    """

    def __init__(self, meta_graph: MetaGraph, bindings: "DomainBindings"):
        """Initialize instantiated graph.

        Args:
            meta_graph: The abstract meta-graph
            bindings: Domain-specific operator bindings
        """
        self.meta_graph = meta_graph
        self.bindings = bindings
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        """Ensure all roles in meta-graph have bindings."""
        missing = []
        for role_value in self.meta_graph.nodes:
            # Try AbstractRole first, then str-based check
            try:
                role = AbstractRole(role_value)
                if not self.bindings.has_role(role):
                    missing.append(role_value)
            except ValueError:
                # Non-optimization role: check string-based binding
                if not self.bindings.has_role(role_value):
                    missing.append(role_value)

        if missing:
            raise ValueError(
                f"Domain '{self.bindings.domain}' missing bindings for roles: {missing}"
            )

    @property
    def domain(self) -> str:
        """Get the domain of this instantiation."""
        return self.bindings.domain

    def get_operators_for_role(self, role: Union[str, AbstractRole]) -> list[str]:
        """Get all operators bound to a role."""
        return self.bindings.get_operators_for_role(role)

    def get_primary_operator(self, role: Union[str, AbstractRole]) -> Optional[str]:
        """Get the primary (highest priority) operator for a role."""
        return self.bindings.get_primary_operator(role)

    def select_operator(
        self,
        role: Union[str, AbstractRole],
        problem_size: Optional[int] = None,
        mode: str = "uniform"
    ) -> str:
        """Select an operator for a role.

        Args:
            role: The abstract role (str or AbstractRole)
            problem_size: Optional problem size (reserved for future use)
            mode: Selection mode ("uniform", "primary", "weighted")

        Returns:
            Concrete operator ID

        Raises:
            ValueError: If no operator bound to role
        """
        op = self.bindings.select_operator(role, problem_size, mode)
        if op is None:
            role_str = _role_to_str(role)
            raise ValueError(f"No operator bound for role {role_str}")
        return op

    def get_edge(self, source: Union[str, AbstractRole],
                 target: Union[str, AbstractRole]) -> Optional[MetaEdge]:
        """Get edge from meta-graph."""
        return self.meta_graph.get_edge(source, target)

    def get_outgoing_edges(self, role: Union[str, AbstractRole]) -> list[MetaEdge]:
        """Get outgoing edges from meta-graph."""
        return self.meta_graph.get_outgoing_edges(role)

    def get_successors(self, role: Union[str, AbstractRole]) -> list[str]:
        """Get successor roles as strings."""
        return self.meta_graph.get_successors(role)

    def get_successors_as_roles(self, role: AbstractRole) -> list[AbstractRole]:
        """Get successor roles as AbstractRole enums (backward compat)."""
        return self.meta_graph.get_successors_as_roles(role)

    def get_entry_roles(self) -> list[str]:
        """Get entry roles (generalized from get_construction_roles)."""
        return self.meta_graph.get_entry_roles()

    def get_construction_roles(self) -> list[AbstractRole]:
        """Get construction roles (backward compat)."""
        return self.meta_graph.get_construction_roles()

    def __repr__(self) -> str:
        return (
            f"InstantiatedGraph("
            f"meta={self.meta_graph.name}, "
            f"domain={self.domain}, "
            f"roles={len(self.meta_graph.nodes)}"
            f")"
        )
