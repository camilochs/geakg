"""Transfer Manager: Orchestrate cross-domain transfer learning.

Carga operadores desde refined_pool.json y reglas simbólicas desde akg_snapshot.json,
luego los adapta para un dominio target.

Ejemplo:
    manager = TransferManager()
    result = manager.transfer(
        session_dir="experiments/iterative/20260118_174429_iterative",
        target_domain="vrp",
        target_instance=vrp_instance,
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

from src.geakg.bindings import BindingRegistry, DomainBindings
from src.geakg.roles import AbstractRole
from .adapter import DomainAdapter, AdaptedOperator
from .adapters.vrp_adapter import VRPAdapter
from .adapters.jssp_adapter import JSSPAdapter
from .adapters.pfsp_adapter import PFSPAdapter
from .adapters.lop_adapter import LOPAdapter
from .adapters.qap_adapter import QAPAdapter
from .adapters.sop_adapter import SOPAdapter


@dataclass
class TransferResult:
    """Result of transfer learning operation."""

    source_domain: str
    target_domain: str
    operators_transferred: int
    adapted_operators: list[AdaptedOperator]
    snapshot: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.operators_transferred > 0 and len(self.errors) == 0


class TransferManager:
    """Orchestrate transfer learning between domains.

    Flujo:
    1. Cargar refined_pool.json (operadores con código)
    2. Cargar akg_snapshot.json (pheromones, metagraph, reglas)
    3. Crear adaptador para el dominio target
    4. Adaptar cada operador usando el adaptador
    """

    _adapters: dict[tuple[str, str], Type[DomainAdapter]] = {
        ("tsp", "vrp"): VRPAdapter,
        ("tsp", "jssp"): JSSPAdapter,
        ("tsp", "pfsp"): PFSPAdapter,
        ("tsp", "lop"): LOPAdapter,
        ("tsp", "qap"): QAPAdapter,
        ("tsp", "sop"): SOPAdapter,
    }

    def __init__(self):
        self._compiled_operators: dict[str, callable] = {}

    @classmethod
    def register_adapter(
        cls,
        source_domain: str,
        target_domain: str,
        adapter_class: Type[DomainAdapter],
    ) -> None:
        cls._adapters[(source_domain, target_domain)] = adapter_class

    @classmethod
    def get_supported_transfers(cls) -> list[tuple[str, str]]:
        return list(cls._adapters.keys())

    def transfer(
        self,
        session_dir: str,
        target_domain: str,
        target_instance: Any,
        target_registry: BindingRegistry | None = None,
    ) -> TransferResult:
        """Transfer from a training session directory.

        Args:
            session_dir: Path to session dir (contains refined_pool.json, akg_snapshot.json)
            target_domain: Target domain name
            target_instance: Instance from target domain
            target_registry: Optional BindingRegistry to update

        Returns:
            TransferResult with adapted operators
        """
        session_path = Path(session_dir)
        pool_path = session_path / "refined_pool.json"
        snapshot_path = session_path / "akg_snapshot.json"

        errors = []
        warnings = []

        # Load pool
        if not pool_path.exists():
            return TransferResult(
                source_domain="unknown",
                target_domain=target_domain,
                operators_transferred=0,
                adapted_operators=[],
                errors=[f"Pool not found: {pool_path}"],
            )

        with open(pool_path) as f:
            pool_data = json.load(f)

        # Load snapshot (optional but recommended)
        snapshot = {}
        if snapshot_path.exists():
            with open(snapshot_path) as f:
                snapshot = json.load(f)
        else:
            warnings.append(f"Snapshot not found: {snapshot_path}")

        # Determine source domain
        source_domain = pool_data.get("metadata", {}).get("domain", "tsp")

        # Check adapter exists
        adapter_key = (source_domain, target_domain)
        if adapter_key not in self._adapters:
            return TransferResult(
                source_domain=source_domain,
                target_domain=target_domain,
                operators_transferred=0,
                adapted_operators=[],
                errors=[f"No adapter for {source_domain} -> {target_domain}"],
            )

        # Create adapter
        adapter_class = self._adapters[adapter_key]
        adapter = adapter_class(target_instance)

        # Get operators from pool
        operators_by_role = pool_data.get("operators_by_role", {})
        adapted_operators = []

        for role_str, ops_list in operators_by_role.items():
            for op_data in ops_list:
                op_id = op_data.get("name", op_data.get("id", ""))
                code = op_data.get("code", "")
                design_choices = op_data.get("design_choices", {})

                if not code:
                    warnings.append(f"No code for operator {op_id}")
                    continue

                # Check if this is a LLaMEA operator (has solve_tsp signature)
                is_llamea = (
                    design_choices.get("type") == "llamea_evolved" or
                    "llamea" in op_id.lower() or
                    "def solve_tsp" in code
                )

                # Compile operator
                compiled_fn = self._compile_operator(code, op_id, is_llamea=is_llamea)
                if compiled_fn is None:
                    warnings.append(f"Failed to compile operator {op_id}")
                    continue

                # Parse role
                try:
                    role = AbstractRole(role_str)
                except ValueError:
                    warnings.append(f"Unknown role '{role_str}' for operator {op_id}")
                    continue

                # Adapt operator (with special handling for LLaMEA)
                try:
                    if is_llamea:
                        # LLaMEA operators have signature: solve_tsp(distance_matrix) -> tour
                        # Wrap to use the adapter's source context distance matrix
                        adapted = self._adapt_llamea_operator(
                            adapter=adapter,
                            operator_id=op_id,
                            solve_tsp_fn=compiled_fn,
                            role=role,
                            original_code=code,
                        )
                    else:
                        adapted = adapter.adapt_operator(
                            operator_id=op_id,
                            operator_fn=compiled_fn,
                            role=role,
                            original_code=code,
                        )
                    adapted_operators.append(adapted)
                    self._compiled_operators[adapted.operator_id] = adapted.adapted_fn
                except Exception as e:
                    warnings.append(f"Failed to adapt {op_id}: {e}")

        # Register in target registry if provided
        if target_registry is not None and adapted_operators:
            bindings = target_registry.get_domain(target_domain)
            if bindings is None:
                bindings = DomainBindings(domain=target_domain)
                target_registry.register_domain(bindings)

            for adapted in adapted_operators:
                bindings.add_operator(
                    operator_id=adapted.operator_id,
                    role=adapted.role,
                    weight=adapted.weight,
                )

        return TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            operators_transferred=len(adapted_operators),
            adapted_operators=adapted_operators,
            snapshot=snapshot,
            errors=errors,
            warnings=warnings,
        )

    def transfer_from_akg(
        self,
        source_snapshot: str,
        target_domain: str,
        target_instance: Any,
        target_registry: BindingRegistry | None = None,
    ) -> TransferResult:
        """Transfer from snapshot path (legacy API).

        Infers session_dir from snapshot path.
        """
        snapshot_path = Path(source_snapshot)
        session_dir = snapshot_path.parent
        return self.transfer(
            session_dir=str(session_dir),
            target_domain=target_domain,
            target_instance=target_instance,
            target_registry=target_registry,
        )

    def _compile_operator(
        self, code: str, operator_id: str, is_llamea: bool = False
    ) -> callable | None:
        """Compile operator code to callable."""
        namespace = {
            "math": __import__("math"),
            "random": __import__("random"),
            "time": __import__("time"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "collections": __import__("collections"),
        }

        try:
            exec(code, namespace)
        except Exception:
            return None

        # For LLaMEA operators, look for solve_tsp specifically
        if is_llamea and "solve_tsp" in namespace and callable(namespace["solve_tsp"]):
            return namespace["solve_tsp"]

        # Find the function
        possible_names = [operator_id, "operator", "apply", "run"]
        for name in possible_names:
            if name in namespace and callable(namespace[name]):
                return namespace[name]

        # Find any callable
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                if not hasattr(obj, "__module__") or obj.__module__ != "builtins":
                    return obj

        return None

    def _adapt_llamea_operator(
        self,
        adapter: "DomainAdapter",
        operator_id: str,
        solve_tsp_fn: callable,
        role: "AbstractRole",
        original_code: str = "",
    ) -> "AdaptedOperator":
        """Adapt a LLaMEA operator for cross-domain transfer.

        LLaMEA operators have signature: solve_tsp(distance_matrix) -> tour
        We wrap them to:
        1. Get the pseudo distance_matrix from the adapter's SourceContext
        2. Call solve_tsp(pseudo_dm) to get a tour
        3. Convert the tour back to target domain representation

        This allows the sophisticated TSP algorithms evolved by LLaMEA
        to be applied to QAP, JSSP, etc. via the domain adapters.
        """
        context = adapter.get_context()

        def adapted_fn(target_solution, target_instance=None):
            """Wrapped LLaMEA operator for target domain."""
            try:
                # Get pseudo distance matrix from adapter context
                dm = context.distance_matrix

                # Call LLaMEA's solve_tsp with the pseudo distance matrix
                # This generates a complete tour optimized for the pseudo-TSP
                tour = solve_tsp_fn(dm)

                # Validate tour is a valid permutation
                n = len(dm)
                if not isinstance(tour, list) or len(tour) != n or set(tour) != set(range(n)):
                    return target_solution

                # Convert tour back to target domain representation
                result = adapter.from_source_repr(tour, context)

                # Validate result in target domain
                if adapter.validate_result(result):
                    return result

                return target_solution

            except Exception:
                return target_solution

        return AdaptedOperator(
            operator_id=f"{operator_id}_{adapter.target_domain}",
            original_id=operator_id,
            role=role,
            source_domain=adapter.source_domain,
            target_domain=adapter.target_domain,
            weight=1.0,
            description=f"LLaMEA operator adapted from TSP: {operator_id}",
            adapted_fn=adapted_fn,
            original_code=original_code,
        )

    def get_compiled_operator(self, operator_id: str) -> callable | None:
        return self._compiled_operators.get(operator_id)

    def get_all_compiled_operators(self) -> dict[str, callable]:
        return self._compiled_operators.copy()


def transfer_tsp_to_vrp(snapshot_path: str, vrp_instance: Any) -> TransferResult:
    """Convenience function for TSP→VRP transfer."""
    manager = TransferManager()
    return manager.transfer_from_akg(
        source_snapshot=snapshot_path,
        target_domain="vrp",
        target_instance=vrp_instance,
    )
