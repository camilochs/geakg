"""Domain Bindings: Map Abstract Roles to Concrete Operators.

This module provides the binding mechanism that connects domain-agnostic
Abstract Roles to domain-specific concrete operators.

Key insight: The same MetaGraph (LLM-generated) can be instantiated for
different domains by swapping bindings. This is explicit transfer learning
through abstraction, not "hidden dictionary" hacks.

NS-SE PURE MODE:
By default, only generic operators (21 for permutations) are used.
Domain-specific operators are DISABLED. dynamic mode synthesizes specialized
operators when needed.

Generalization: role fields accept both AbstractRole and plain str.
This enables non-optimization case studies (NAS, etc.) to use the
same binding infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union
import random

from .roles import AbstractRole, RoleCategory, ROLE_CATALOG

if TYPE_CHECKING:
    from .representations import RepresentationType, DomainSpec


def _role_key(role: Union[str, AbstractRole]) -> str:
    """Normalize a role to its string key."""
    if isinstance(role, AbstractRole):
        return role.value
    return role


@dataclass
class OperatorBinding:
    """Binds a concrete operator to an abstract role.

    Attributes:
        operator_id: The concrete operator identifier
        role: The abstract role this operator implements (str or AbstractRole)
        domain: The domain this binding applies to (e.g., "tsp", "jssp")
        priority: Selection priority (higher = preferred), default 1
        weight: Selection weight for weighted random selection
        description: Human-readable description of this binding
    """

    operator_id: str
    role: Union[str, AbstractRole]
    domain: str
    priority: int = 1
    weight: float = 1.0
    description: str = ""

    def __post_init__(self):
        if not self.description:
            role_str = _role_key(self.role)
            self.description = f"{self.operator_id} implements {role_str}"

    @property
    def role_key(self) -> str:
        return _role_key(self.role)


@dataclass
class DomainBindings:
    """Collection of operator bindings for a specific domain.

    Provides methods to select operators for roles, supporting both
    deterministic (priority-based) and stochastic (weight-based) selection.

    bindings dict is keyed by string role IDs for generality.
    """

    domain: str
    bindings: dict[str, list[OperatorBinding]] = field(default_factory=dict)

    def add_binding(self, binding: OperatorBinding) -> None:
        """Add an operator binding."""
        key = binding.role_key
        if key not in self.bindings:
            self.bindings[key] = []
        self.bindings[key].append(binding)
        # Keep sorted by priority (descending)
        self.bindings[key].sort(key=lambda b: -b.priority)

    def get_operators_for_role(self, role: Union[str, AbstractRole]) -> list[str]:
        """Get all operators bound to a role."""
        key = _role_key(role)
        if key not in self.bindings:
            return []
        return [b.operator_id for b in self.bindings[key]]

    def clear_operators_for_role(self, role: Union[str, AbstractRole]) -> None:
        """Remove all operator bindings for a role."""
        key = _role_key(role)
        if key in self.bindings:
            self.bindings[key] = []

    def get_primary_operator(self, role: Union[str, AbstractRole]) -> Optional[str]:
        """Get the highest-priority operator for a role."""
        key = _role_key(role)
        if key not in self.bindings or not self.bindings[key]:
            return None
        return self.bindings[key][0].operator_id

    def select_operator(
        self,
        role: Union[str, AbstractRole],
        problem_size: Optional[int] = None,
        mode: str = "uniform"
    ) -> Optional[str]:
        """Select an operator for a role.

        Args:
            role: The abstract role to select for (str or AbstractRole)
            problem_size: Optional problem size (reserved for future use)
            mode: Selection mode:
                - "uniform": Uniform random selection (DEFAULT)
                - "primary": Always select highest-priority operator
                - "weighted": Weighted random by operator weights

        Returns:
            Operator ID or None if no binding exists
        """
        key = _role_key(role)
        if key not in self.bindings or not self.bindings[key]:
            return None

        candidates = self.bindings[key]

        if len(candidates) == 1:
            return candidates[0].operator_id

        if mode == "primary":
            return candidates[0].operator_id

        elif mode == "uniform":
            return random.choice(candidates).operator_id

        elif mode == "weighted":
            total_weight = sum(b.weight for b in candidates)
            if total_weight == 0:
                return candidates[0].operator_id

            r = random.random() * total_weight
            cumsum = 0.0
            for b in candidates:
                cumsum += b.weight
                if r <= cumsum:
                    return b.operator_id
            return candidates[-1].operator_id

        else:
            return random.choice(candidates).operator_id

    def get_operator_weight(self, role: Union[str, AbstractRole], operator_id: str) -> float:
        """Get the weight for a specific operator in a role."""
        key = _role_key(role)
        if key not in self.bindings:
            return 1.0
        for binding in self.bindings[key]:
            if binding.operator_id == operator_id:
                return binding.weight
        return 1.0

    def has_role(self, role: Union[str, AbstractRole]) -> bool:
        """Check if this domain has bindings for a role."""
        key = _role_key(role)
        return key in self.bindings and len(self.bindings[key]) > 0

    def get_bound_roles(self) -> list[str]:
        """Get all roles that have at least one binding (as strings)."""
        return [r for r in self.bindings if self.bindings[r]]

    def add_synthesized_operator(
        self,
        operator_id: str,
        role: Union[str, AbstractRole],
        weight: float = 1.0,
        description: str = "",
    ) -> OperatorBinding:
        """Add a dynamically synthesized operator binding."""
        binding = OperatorBinding(
            operator_id=operator_id,
            role=role,
            domain=self.domain,
            priority=1,
            weight=weight,
            description=description or f"Synthesized operator: {operator_id}",
        )
        self.add_binding(binding)
        return binding

    def __repr__(self) -> str:
        roles = len(self.get_bound_roles())
        total_ops = sum(len(ops) for ops in self.bindings.values())
        return f"DomainBindings(domain={self.domain}, roles={roles}, operators={total_ops})"


class BindingRegistry:
    """Global registry of domain bindings.

    NS-SE PURE MODE (default):
    - Only generic operators (21 for permutations) are loaded
    - Domain-specific operators are DISABLED
    """

    _instance: Optional["BindingRegistry"] = None
    _domains: dict[str, DomainBindings]
    _domain_specs: dict[str, "DomainSpec"]
    use_domain_specific: bool

    def __new__(cls) -> "BindingRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._domains = {}
            cls._instance._domain_specs = {}
            cls._instance.use_domain_specific = False
            cls._instance._initialize_default_bindings()
        return cls._instance

    def _initialize_default_bindings(self) -> None:
        """Initialize domain bindings."""
        from .representations import RepresentationType, DomainSpec

        self._register_domain_spec(DomainSpec(
            name="tsp",
            representation=RepresentationType.PERMUTATION,
            fitness_fn=lambda tour, inst: 0.0,
            solution_size_fn=lambda inst: getattr(inst, 'num_cities', 100),
        ))
        self._register_domain_spec(DomainSpec(
            name="jssp",
            representation=RepresentationType.PERMUTATION,
            fitness_fn=lambda sched, inst: 0.0,
            solution_size_fn=lambda inst: getattr(inst, 'num_operations', 100),
        ))
        self._register_domain_spec(DomainSpec(
            name="vrp",
            representation=RepresentationType.PERMUTATION,
            fitness_fn=lambda routes, inst: 0.0,
            solution_size_fn=lambda inst: getattr(inst, 'num_customers', 100),
        ))
        self._register_domain_spec(DomainSpec(
            name="bpp",
            representation=RepresentationType.PARTITION,
            fitness_fn=lambda bins, inst: 0.0,
            solution_size_fn=lambda inst: getattr(inst, 'num_items', 100),
        ))

        for domain_name, spec in self._domain_specs.items():
            bindings = self._create_bindings_for_domain(domain_name, spec)
            self._domains[domain_name] = bindings

    def _register_domain_spec(self, spec: "DomainSpec") -> None:
        """Register a domain specification."""
        self._domain_specs[spec.name] = spec

    def _create_bindings_for_domain(
        self,
        domain_name: str,
        spec: "DomainSpec",
    ) -> DomainBindings:
        """Create bindings for a domain using generic operators."""
        from .representations import RepresentationType
        from .generic_operators import get_operators_for_representation

        bindings = DomainBindings(domain=domain_name)

        try:
            generic_ops = get_operators_for_representation(spec.representation)
            for op in generic_ops.get_all_operators():
                try:
                    role = AbstractRole(op.role)
                except ValueError:
                    continue

                bindings.add_binding(OperatorBinding(
                    operator_id=op.operator_id,
                    role=role,
                    domain=domain_name,
                    priority=1,
                    weight=op.weight,
                    description=op.description,
                ))
        except KeyError:
            pass

        if self.use_domain_specific:
            self._add_domain_specific_operators(bindings, domain_name)

        return bindings

    def _add_domain_specific_operators(
        self,
        bindings: DomainBindings,
        domain_name: str,
    ) -> None:
        """Add domain-specific operators (WARM START mode only)."""
        if domain_name == "tsp":
            domain_bindings = create_tsp_bindings()
        elif domain_name == "jssp":
            domain_bindings = create_jssp_bindings()
        elif domain_name == "vrp":
            domain_bindings = create_vrp_bindings()
        elif domain_name == "bpp":
            domain_bindings = create_bpp_bindings()
        else:
            return

        for role_key, role_bindings in domain_bindings.bindings.items():
            for binding in role_bindings:
                bindings.add_binding(binding)

    def register_domain(self, bindings: DomainBindings) -> None:
        """Register bindings for a domain."""
        self._domains[bindings.domain] = bindings

    def register_domain_from_spec(self, spec: "DomainSpec") -> DomainBindings:
        """Register a new domain from its specification."""
        self._domain_specs[spec.name] = spec
        bindings = self._create_bindings_for_domain(spec.name, spec)
        self._domains[spec.name] = bindings
        return bindings

    def get_domain(self, domain: str) -> Optional[DomainBindings]:
        """Get bindings for a domain."""
        return self._domains.get(domain)

    def get_domain_spec(self, domain: str) -> Optional["DomainSpec"]:
        """Get the specification for a domain."""
        return self._domain_specs.get(domain)

    def has_domain(self, domain: str) -> bool:
        """Check if a domain is registered."""
        return domain in self._domains

    def list_domains(self) -> list[str]:
        """List all registered domains."""
        return list(self._domains.keys())

    def get_operator_count(self, domain: str) -> int:
        """Get total number of operators for a domain."""
        bindings = self.get_domain(domain)
        if bindings is None:
            return 0
        return sum(len(ops) for ops in bindings.bindings.values())

    def get_mode(self) -> str:
        """Get current binding mode."""
        return "warm_start" if self.use_domain_specific else "pure_generic"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None


# =============================================================================
# TSP Bindings: operators -> 11 roles
# =============================================================================

def create_tsp_bindings() -> DomainBindings:
    """Create operator bindings for TSP domain."""
    bindings = DomainBindings(domain="tsp")

    # === CONSTRUCTION ===
    bindings.add_binding(OperatorBinding(
        operator_id="greedy_nearest_neighbor",
        role=AbstractRole.CONST_GREEDY,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Classic NN heuristic: always visit closest unvisited city",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="nearest_addition",
        role=AbstractRole.CONST_GREEDY,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Add nearest city to current tour fragment",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="cheapest_insertion",
        role=AbstractRole.CONST_INSERTION,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Insert city at position minimizing tour increase",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="farthest_insertion",
        role=AbstractRole.CONST_INSERTION,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Insert farthest city first, then cheapest position",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="nearest_insertion",
        role=AbstractRole.CONST_INSERTION,
        domain="tsp",
        priority=1,
        weight=0.8,
        description="Insert nearest city to tour at best position",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="savings_heuristic",
        role=AbstractRole.CONST_SAVINGS,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Clarke-Wright savings: merge routes by savings",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="christofides_construction",
        role=AbstractRole.CONST_SAVINGS,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="MST + matching for 1.5-approximation",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="random_insertion",
        role=AbstractRole.CONST_RANDOM,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Random city order, insert at random positions",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="convex_hull_start",
        role=AbstractRole.CONST_GREEDY,
        domain="tsp",
        priority=1,
        weight=0.8,
        description="Start from convex hull of cities",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="cluster_first",
        role=AbstractRole.CONST_SAVINGS,
        domain="tsp",
        priority=1,
        weight=0.9,
        description="Cluster cities first, then route within clusters",
    ))

    # === LOCAL SEARCH ===
    bindings.add_binding(OperatorBinding(
        operator_id="two_opt",
        role=AbstractRole.LS_INTENSIFY_SMALL,
        domain="tsp",
        priority=3,
        weight=2.0,
        description="Remove 2 edges, reconnect in other way",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="swap",
        role=AbstractRole.LS_INTENSIFY_SMALL,
        domain="tsp",
        priority=2,
        weight=1.0,
        description="Swap positions of two cities",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="insert",
        role=AbstractRole.LS_INTENSIFY_SMALL,
        domain="tsp",
        priority=1,
        weight=0.8,
        description="Remove city, reinsert at better position",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="invert",
        role=AbstractRole.LS_INTENSIFY_SMALL,
        domain="tsp",
        priority=1,
        weight=0.6,
        description="Reverse a segment of the tour",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="three_opt",
        role=AbstractRole.LS_INTENSIFY_MEDIUM,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Remove 3 edges, reconnect optimally",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="or_opt",
        role=AbstractRole.LS_INTENSIFY_MEDIUM,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Relocate segment of 1-3 cities",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="relocate",
        role=AbstractRole.LS_INTENSIFY_MEDIUM,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Move city to different position",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="lin_kernighan",
        role=AbstractRole.LS_INTENSIFY_LARGE,
        domain="tsp",
        priority=2,
        weight=2.0,
        description="Variable-depth k-opt with backtracking",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="variable_neighborhood",
        role=AbstractRole.LS_CHAIN,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Cycle through 2-opt, or-opt, 3-opt",
    ))

    # === PERTURBATION ===
    bindings.add_binding(OperatorBinding(
        operator_id="double_bridge",
        role=AbstractRole.PERT_ESCAPE_SMALL,
        domain="tsp",
        priority=2,
        weight=2.0,
        description="Split tour into 4 parts, reconnect differently",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="random_segment_shuffle",
        role=AbstractRole.PERT_ESCAPE_SMALL,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Shuffle a random segment of the tour",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="ruin_recreate",
        role=AbstractRole.PERT_ESCAPE_LARGE,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Remove portion of tour, rebuild with construction",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="large_neighborhood_search",
        role=AbstractRole.PERT_ESCAPE_LARGE,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Destroy-repair with multiple strategies",
    ))

    bindings.add_binding(OperatorBinding(
        operator_id="guided_mutation",
        role=AbstractRole.PERT_ADAPTIVE,
        domain="tsp",
        priority=2,
        weight=1.5,
        description="Perturbation guided by edge frequencies",
    ))
    bindings.add_binding(OperatorBinding(
        operator_id="adaptive_mutation",
        role=AbstractRole.PERT_ADAPTIVE,
        domain="tsp",
        priority=1,
        weight=1.0,
        description="Perturbation strength adapts to search state",
    ))

    return bindings


# =============================================================================
# JSSP Bindings: Job Shop Scheduling operators -> 11 roles
# =============================================================================

def create_jssp_bindings() -> DomainBindings:
    """Create operator bindings for JSSP domain."""
    bindings = DomainBindings(domain="jssp")

    # === CONSTRUCTION ===
    bindings.add_binding(OperatorBinding(operator_id="spt_dispatch", role=AbstractRole.CONST_GREEDY, domain="jssp", priority=2, weight=1.5, description="Shortest Processing Time first"))
    bindings.add_binding(OperatorBinding(operator_id="mwkr_dispatch", role=AbstractRole.CONST_GREEDY, domain="jssp", priority=1, weight=1.0, description="Most Work Remaining first"))
    bindings.add_binding(OperatorBinding(operator_id="fifo_dispatch", role=AbstractRole.CONST_GREEDY, domain="jssp", priority=1, weight=0.8, description="First In First Out dispatch"))
    bindings.add_binding(OperatorBinding(operator_id="est_insertion", role=AbstractRole.CONST_INSERTION, domain="jssp", priority=2, weight=1.5, description="Insert at Earliest Start Time"))
    bindings.add_binding(OperatorBinding(operator_id="ect_insertion", role=AbstractRole.CONST_INSERTION, domain="jssp", priority=1, weight=1.0, description="Insert at Earliest Completion Time"))
    bindings.add_binding(OperatorBinding(operator_id="shifting_bottleneck", role=AbstractRole.CONST_SAVINGS, domain="jssp", priority=1, weight=1.5, description="Iteratively schedule bottleneck machines"))
    bindings.add_binding(OperatorBinding(operator_id="random_dispatch", role=AbstractRole.CONST_RANDOM, domain="jssp", priority=1, weight=1.0, description="Random priority dispatch"))

    # === LOCAL SEARCH ===
    bindings.add_binding(OperatorBinding(operator_id="adjacent_swap", role=AbstractRole.LS_INTENSIFY_SMALL, domain="jssp", priority=2, weight=1.5, description="Swap adjacent operations on critical path"))
    bindings.add_binding(OperatorBinding(operator_id="critical_swap", role=AbstractRole.LS_INTENSIFY_SMALL, domain="jssp", priority=2, weight=1.5, description="Swap operations on critical path"))
    bindings.add_binding(OperatorBinding(operator_id="block_move", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="jssp", priority=2, weight=1.5, description="Move block of operations"))
    bindings.add_binding(OperatorBinding(operator_id="block_swap", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="jssp", priority=1, weight=1.0, description="Swap blocks of operations"))
    bindings.add_binding(OperatorBinding(operator_id="ejection_chain", role=AbstractRole.LS_INTENSIFY_LARGE, domain="jssp", priority=1, weight=1.5, description="Chain of ejections and reinsertions"))
    bindings.add_binding(OperatorBinding(operator_id="path_relinking", role=AbstractRole.LS_INTENSIFY_LARGE, domain="jssp", priority=1, weight=1.0, description="Path between elite solutions"))
    bindings.add_binding(OperatorBinding(operator_id="vns_local", role=AbstractRole.LS_CHAIN, domain="jssp", priority=1, weight=1.0, description="Variable neighborhood search"))
    bindings.add_binding(OperatorBinding(operator_id="sequential_vnd", role=AbstractRole.LS_CHAIN, domain="jssp", priority=1, weight=0.8, description="Sequential VND with multiple moves"))

    # === PERTURBATION ===
    bindings.add_binding(OperatorBinding(operator_id="random_delay", role=AbstractRole.PERT_ESCAPE_SMALL, domain="jssp", priority=2, weight=1.5, description="Randomly delay some operations"))
    bindings.add_binding(OperatorBinding(operator_id="critical_block_shuffle", role=AbstractRole.PERT_ESCAPE_SMALL, domain="jssp", priority=1, weight=1.0, description="Shuffle critical path blocks"))
    bindings.add_binding(OperatorBinding(operator_id="ruin_recreate", role=AbstractRole.PERT_ESCAPE_LARGE, domain="jssp", priority=2, weight=1.5, description="Remove machines, re-schedule"))
    bindings.add_binding(OperatorBinding(operator_id="destroy_repair", role=AbstractRole.PERT_ESCAPE_LARGE, domain="jssp", priority=1, weight=1.0, description="Destroy portion of schedule, repair"))
    bindings.add_binding(OperatorBinding(operator_id="guided_perturbation", role=AbstractRole.PERT_ADAPTIVE, domain="jssp", priority=2, weight=1.5, description="Perturbation guided by bottleneck analysis"))
    bindings.add_binding(OperatorBinding(operator_id="frequency_based_shake", role=AbstractRole.PERT_ADAPTIVE, domain="jssp", priority=1, weight=1.0, description="Shake based on operation frequencies"))

    return bindings


# =============================================================================
# VRP Bindings
# =============================================================================

def create_vrp_bindings() -> DomainBindings:
    """Create operator bindings for VRP domain."""
    bindings = DomainBindings(domain="vrp")

    bindings.add_binding(OperatorBinding(operator_id="nearest_neighbor_vrp", role=AbstractRole.CONST_GREEDY, domain="vrp", priority=2, weight=1.5, description="Nearest neighbor for each route"))
    bindings.add_binding(OperatorBinding(operator_id="nearest_insertion_vrp", role=AbstractRole.CONST_GREEDY, domain="vrp", priority=1, weight=1.0, description="Insert nearest unrouted customer"))
    bindings.add_binding(OperatorBinding(operator_id="sweep_construction", role=AbstractRole.CONST_GREEDY, domain="vrp", priority=1, weight=0.8, description="Sweep algorithm: angular sorting from depot"))
    bindings.add_binding(OperatorBinding(operator_id="cheapest_insertion_vrp", role=AbstractRole.CONST_INSERTION, domain="vrp", priority=2, weight=1.5, description="Insert at minimum cost position"))
    bindings.add_binding(OperatorBinding(operator_id="farthest_insertion_vrp", role=AbstractRole.CONST_INSERTION, domain="vrp", priority=1, weight=1.0, description="Insert farthest customer first"))
    bindings.add_binding(OperatorBinding(operator_id="regret_insertion", role=AbstractRole.CONST_INSERTION, domain="vrp", priority=1, weight=0.8, description="Regret-k insertion heuristic"))
    bindings.add_binding(OperatorBinding(operator_id="savings_vrp", role=AbstractRole.CONST_SAVINGS, domain="vrp", priority=2, weight=1.5, description="Clarke-Wright savings algorithm"))
    bindings.add_binding(OperatorBinding(operator_id="parallel_savings", role=AbstractRole.CONST_SAVINGS, domain="vrp", priority=1, weight=1.0, description="Parallel version of savings"))
    bindings.add_binding(OperatorBinding(operator_id="random_vrp", role=AbstractRole.CONST_RANDOM, domain="vrp", priority=1, weight=1.0, description="Random feasible route construction"))

    bindings.add_binding(OperatorBinding(operator_id="two_opt_vrp", role=AbstractRole.LS_INTENSIFY_SMALL, domain="vrp", priority=2, weight=1.5, description="2-opt within routes"))
    bindings.add_binding(OperatorBinding(operator_id="swap_within", role=AbstractRole.LS_INTENSIFY_SMALL, domain="vrp", priority=2, weight=1.5, description="Swap customers within same route"))
    bindings.add_binding(OperatorBinding(operator_id="relocate_within", role=AbstractRole.LS_INTENSIFY_SMALL, domain="vrp", priority=1, weight=1.0, description="Relocate customer within route"))
    bindings.add_binding(OperatorBinding(operator_id="swap_between", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="vrp", priority=2, weight=1.5, description="Swap customers between routes"))
    bindings.add_binding(OperatorBinding(operator_id="relocate_between", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="vrp", priority=2, weight=1.5, description="Move customer to different route"))
    bindings.add_binding(OperatorBinding(operator_id="or_opt_vrp", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="vrp", priority=1, weight=1.0, description="Or-opt: move segment of customers"))
    bindings.add_binding(OperatorBinding(operator_id="cross_exchange", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="vrp", priority=1, weight=0.8, description="Exchange segments between routes"))
    bindings.add_binding(OperatorBinding(operator_id="lin_kernighan_vrp", role=AbstractRole.LS_INTENSIFY_LARGE, domain="vrp", priority=1, weight=1.5, description="LK-style variable depth search"))
    bindings.add_binding(OperatorBinding(operator_id="ejection_chain_vrp", role=AbstractRole.LS_INTENSIFY_LARGE, domain="vrp", priority=1, weight=1.0, description="Cyclic ejection chains"))
    bindings.add_binding(OperatorBinding(operator_id="vns_vrp", role=AbstractRole.LS_CHAIN, domain="vrp", priority=1, weight=1.0, description="VNS cycling through neighborhoods"))
    bindings.add_binding(OperatorBinding(operator_id="sequential_vnd_vrp", role=AbstractRole.LS_CHAIN, domain="vrp", priority=1, weight=0.8, description="Sequential VND for VRP"))

    bindings.add_binding(OperatorBinding(operator_id="random_removal", role=AbstractRole.PERT_ESCAPE_SMALL, domain="vrp", priority=2, weight=1.5, description="Remove random customers, reinsert"))
    bindings.add_binding(OperatorBinding(operator_id="worst_removal", role=AbstractRole.PERT_ESCAPE_SMALL, domain="vrp", priority=1, weight=1.0, description="Remove expensive customers"))
    bindings.add_binding(OperatorBinding(operator_id="ruin_recreate_vrp", role=AbstractRole.PERT_ESCAPE_LARGE, domain="vrp", priority=2, weight=1.5, description="Large ruin and recreate"))
    bindings.add_binding(OperatorBinding(operator_id="route_destruction", role=AbstractRole.PERT_ESCAPE_LARGE, domain="vrp", priority=1, weight=1.0, description="Destroy entire routes, rebuild"))
    bindings.add_binding(OperatorBinding(operator_id="shaw_removal", role=AbstractRole.PERT_ADAPTIVE, domain="vrp", priority=2, weight=1.5, description="Remove related customers (Shaw)"))
    bindings.add_binding(OperatorBinding(operator_id="historic_removal", role=AbstractRole.PERT_ADAPTIVE, domain="vrp", priority=1, weight=1.0, description="Remove based on historical performance"))

    return bindings


# =============================================================================
# BPP Bindings
# =============================================================================

def create_bpp_bindings() -> DomainBindings:
    """Create operator bindings for BPP domain."""
    bindings = DomainBindings(domain="bpp")

    bindings.add_binding(OperatorBinding(operator_id="first_fit", role=AbstractRole.CONST_GREEDY, domain="bpp", priority=2, weight=1.5, description="First Fit: place in first bin that fits"))
    bindings.add_binding(OperatorBinding(operator_id="best_fit", role=AbstractRole.CONST_GREEDY, domain="bpp", priority=2, weight=1.5, description="Best Fit: place in tightest bin"))
    bindings.add_binding(OperatorBinding(operator_id="worst_fit", role=AbstractRole.CONST_GREEDY, domain="bpp", priority=1, weight=0.8, description="Worst Fit: place in loosest bin"))
    bindings.add_binding(OperatorBinding(operator_id="first_fit_decreasing", role=AbstractRole.CONST_INSERTION, domain="bpp", priority=2, weight=1.5, description="FFD: sort by size, then first fit"))
    bindings.add_binding(OperatorBinding(operator_id="best_fit_decreasing", role=AbstractRole.CONST_INSERTION, domain="bpp", priority=2, weight=1.5, description="BFD: sort by size, then best fit"))
    bindings.add_binding(OperatorBinding(operator_id="next_fit_decreasing", role=AbstractRole.CONST_INSERTION, domain="bpp", priority=1, weight=1.0, description="NFD: sort by size, then next fit"))
    bindings.add_binding(OperatorBinding(operator_id="djang_fitch", role=AbstractRole.CONST_SAVINGS, domain="bpp", priority=1, weight=1.5, description="Djang-Fitch: MTP-based construction"))
    bindings.add_binding(OperatorBinding(operator_id="best_k_fit", role=AbstractRole.CONST_SAVINGS, domain="bpp", priority=1, weight=1.0, description="Best-k-fit heuristic"))
    bindings.add_binding(OperatorBinding(operator_id="random_fit", role=AbstractRole.CONST_RANDOM, domain="bpp", priority=1, weight=1.0, description="Random feasible placement"))

    bindings.add_binding(OperatorBinding(operator_id="swap_items", role=AbstractRole.LS_INTENSIFY_SMALL, domain="bpp", priority=2, weight=1.5, description="Swap items between bins"))
    bindings.add_binding(OperatorBinding(operator_id="move_item", role=AbstractRole.LS_INTENSIFY_SMALL, domain="bpp", priority=2, weight=1.5, description="Move item to different bin"))
    bindings.add_binding(OperatorBinding(operator_id="swap_pairs", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="bpp", priority=2, weight=1.5, description="Swap pairs of items"))
    bindings.add_binding(OperatorBinding(operator_id="chain_move", role=AbstractRole.LS_INTENSIFY_MEDIUM, domain="bpp", priority=1, weight=1.0, description="Chain of item movements"))
    bindings.add_binding(OperatorBinding(operator_id="reduce_bins", role=AbstractRole.LS_INTENSIFY_LARGE, domain="bpp", priority=2, weight=1.5, description="Try to reduce number of bins"))
    bindings.add_binding(OperatorBinding(operator_id="bin_completion", role=AbstractRole.LS_INTENSIFY_LARGE, domain="bpp", priority=1, weight=1.0, description="Try to complete partially filled bins"))
    bindings.add_binding(OperatorBinding(operator_id="vns_bpp", role=AbstractRole.LS_CHAIN, domain="bpp", priority=1, weight=1.0, description="Variable neighborhood search for BPP"))
    bindings.add_binding(OperatorBinding(operator_id="sequential_vnd_bpp", role=AbstractRole.LS_CHAIN, domain="bpp", priority=1, weight=0.8, description="Sequential VND for BPP"))

    bindings.add_binding(OperatorBinding(operator_id="random_reassign", role=AbstractRole.PERT_ESCAPE_SMALL, domain="bpp", priority=2, weight=1.5, description="Randomly reassign some items"))
    bindings.add_binding(OperatorBinding(operator_id="shuffle_bin", role=AbstractRole.PERT_ESCAPE_SMALL, domain="bpp", priority=1, weight=1.0, description="Shuffle items in selected bins"))
    bindings.add_binding(OperatorBinding(operator_id="empty_bins", role=AbstractRole.PERT_ESCAPE_LARGE, domain="bpp", priority=2, weight=1.5, description="Empty bins and repack"))
    bindings.add_binding(OperatorBinding(operator_id="ruin_recreate_bpp", role=AbstractRole.PERT_ESCAPE_LARGE, domain="bpp", priority=1, weight=1.0, description="Ruin and recreate for BPP"))
    bindings.add_binding(OperatorBinding(operator_id="guided_reassignment", role=AbstractRole.PERT_ADAPTIVE, domain="bpp", priority=2, weight=1.5, description="Reassign based on bin utilization"))
    bindings.add_binding(OperatorBinding(operator_id="frequency_repack", role=AbstractRole.PERT_ADAPTIVE, domain="bpp", priority=1, weight=1.0, description="Repack based on solution history"))

    return bindings


# =============================================================================
# Utility Functions
# =============================================================================

def get_binding_stats(domain: str) -> dict:
    """Get statistics about bindings for a domain."""
    registry = BindingRegistry()
    bindings = registry.get_domain(domain)

    if not bindings:
        return {"domain": domain, "error": "Domain not found"}

    roles_bound = bindings.get_bound_roles()
    operators_per_role = {
        role: len(bindings.get_operators_for_role(role))
        for role in roles_bound
    }

    return {
        "domain": domain,
        "total_roles_bound": len(roles_bound),
        "total_roles_available": len(AbstractRole),
        "coverage": len(roles_bound) / len(AbstractRole),
        "operators_per_role": operators_per_role,
        "total_operators": sum(operators_per_role.values()),
    }


def print_binding_table(domain: str) -> None:
    """Print a human-readable table of bindings for a domain."""
    registry = BindingRegistry()
    bindings = registry.get_domain(domain)

    if not bindings:
        print(f"Domain '{domain}' not found")
        return

    print(f"\n{'='*70}")
    print(f"Bindings for domain: {domain.upper()}")
    print(f"{'='*70}")

    for role in AbstractRole:
        ops = bindings.get_operators_for_role(role)
        primary = bindings.get_primary_operator(role)
        if ops:
            print(f"\n{role.value}:")
            for op in ops:
                marker = " [PRIMARY]" if op == primary else ""
                print(f"  - {op}{marker}")
        else:
            print(f"\n{role.value}: (no bindings)")

    print(f"\n{'='*70}")
