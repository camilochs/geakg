"""L1 Synthesis Hook - Lightweight hook for pre-compiled L1 operators.

This hook provides the same interface as DynamicSynthesisHook but without
the synthesis infrastructure. It contains pre-compiled operators from the
L1 pool that can be executed by the ACO.

Key difference from DynamicSynthesisHook:
- No LLM calls - operators are pre-generated
- No trigger logic - no synthesis happens at runtime
- Just a registry of compiled operators from L1 pool
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from src.geakg.execution import compile_operator_code

if TYPE_CHECKING:
    from src.geakg.bindings import DomainBindings
    from src.geakg.layers.l1.pool import OperatorPool


@dataclass
class L1OperatorRecord:
    """Record for an L1 operator."""
    operator_id: str
    role: str
    code: str
    compiled_fn: Callable | None = None


class L1Registry:
    """Simple registry for L1 operators."""

    def __init__(self) -> None:
        self._operators: dict[str, L1OperatorRecord] = {}

    def register(self, record: L1OperatorRecord) -> None:
        """Register an operator."""
        self._operators[record.operator_id] = record

    def get(self, operator_id: str) -> L1OperatorRecord | None:
        """Get operator record by ID."""
        return self._operators.get(operator_id)

    def __contains__(self, operator_id: str) -> bool:
        return operator_id in self._operators

    def __len__(self) -> int:
        return len(self._operators)


class L1SynthesisHook:
    """Lightweight synthesis hook for L1 pool operators.

    Provides the same interface as DynamicSynthesisHook but without
    the synthesis infrastructure. Operators are pre-compiled from the
    L1 pool.

    Usage:
        pool = OperatorPool.load("pools/tsp.json")
        hook = L1SynthesisHook(pool)
        selector = MetaACOSelector(graph, config, synthesis_hook=hook)
    """

    def __init__(self, pool: "OperatorPool") -> None:
        """Initialize hook with L1 pool.

        Args:
            pool: L1 operator pool with operators by role.
        """
        self.pool = pool
        self.registry = L1Registry()
        self._compile_pool_operators()

    def _compile_pool_operators(self) -> None:
        """Compile all operators from the pool."""
        compiled_count = 0
        failed_count = 0

        for role in self.pool.roles:
            operators = self.pool.get_operators_for_role(role)
            for op in operators:
                compiled_fn = compile_operator_code(op.code)
                if compiled_fn is not None:
                    record = L1OperatorRecord(
                        operator_id=op.name,
                        role=role,
                        code=op.code,
                        compiled_fn=compiled_fn,
                    )
                    self.registry.register(record)
                    compiled_count += 1
                else:
                    logger.warning(f"[L1-HOOK] Failed to compile operator: {op.name}")
                    failed_count += 1

        logger.info(
            f"[L1-HOOK] Compiled {compiled_count} operators "
            f"({failed_count} failed) from L0 pool"
        )

    def get_compiled_operator(self, operator_id: str) -> Callable | None:
        """Get the compiled function for an operator.

        Args:
            operator_id: Operator name/ID.

        Returns:
            Compiled function or None if not found.
        """
        record = self.registry.get(operator_id)
        if record is None:
            return None
        return record.compiled_fn

    def on_operator_selected(self, operator_id: str) -> None:
        """Called when an operator is successfully used.

        For L0, this is a no-op (no statistics tracking needed).
        """
        pass

    def check_and_synthesize(self, *args: Any, **kwargs: Any) -> None:
        """No-op: L0 doesn't synthesize at runtime."""
        pass

    def has_pending_syntheses(self) -> bool:
        """No pending syntheses in L0 mode."""
        return False

    def collect_completed_syntheses(self) -> list:
        """No syntheses to collect in L0 mode."""
        return []

    def reset(self) -> None:
        """Reset is a no-op for L0 - operators are fixed."""
        pass

    def register_operators_to_bindings(
        self,
        bindings: "DomainBindings",
        clear_existing: bool = True,
    ) -> None:
        """Register L1 operators to domain bindings.

        This allows the ACO to select L1 operators when choosing
        operators for roles. Uses string-based role keys for generality
        (works with both optimization and NAS roles).

        Args:
            bindings: Domain bindings to add operators to.
            clear_existing: If True, remove existing operators before adding L1 ops.
        """
        from src.geakg.bindings import OperatorBinding

        # Clear existing operators if requested
        if clear_existing:
            roles_to_clear = set()
            for record in self.registry._operators.values():
                roles_to_clear.add(record.role)

            for role_str in roles_to_clear:
                bindings.clear_operators_for_role(role_str)

            logger.debug(f"[L1-HOOK] Cleared existing operators for {len(roles_to_clear)} roles")

        added_count = 0
        for record in self.registry._operators.values():
            # Use string role directly (generalized)
            binding = OperatorBinding(
                operator_id=record.operator_id,
                role=record.role,  # str role key
                domain=bindings.domain,
                priority=10,
                weight=2.0,
                description=f"L1 operator: {record.operator_id}",
            )
            bindings.add_binding(binding)
            added_count += 1

        logger.info(f"[L1-HOOK] Registered {added_count} operators to bindings")


# Backward compatibility aliases
L0OperatorRecord = L1OperatorRecord
L0Registry = L1Registry
L0SynthesisHook = L1SynthesisHook
