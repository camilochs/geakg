"""Domain Adapter: Base class for cross-domain transfer learning.

Implementa el patrón Adapter siguiendo principios de Lampson:
- "Keep Secrets": Detalles del dominio fuente ocultos en to_source_repr()
- "Keep Secrets": Detalles del dominio target ocultos en from_source_repr()
- "Minimum Essentials": Interfaz mínima para adaptación

Ejemplo de flujo TSP→VRP:
    adapter = VRPAdapter(vrp_instance)

    # VRP routes → TSP giant tour
    tsp_tour, context = adapter.to_source_repr(vrp_routes)

    # Apply TSP operator
    improved_tour = tsp_operator(tsp_tour, context.distance_matrix)

    # TSP result → VRP routes
    improved_routes = adapter.from_source_repr(improved_tour)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..roles import AbstractRole


class TransferContext:
    """Context object compatible with synthesized operator interface.

    Provides the ctx methods expected by domain-agnostic operators:
    - ctx.evaluate(solution) -> float  (REQUIRED - injected by adapter)
    - ctx.valid(solution) -> bool

    The evaluate function is injected by the adapter to compute
    the actual cost in the TARGET domain, not TSP.
    """

    def __init__(
        self,
        distance_matrix: list[list[float]],
        evaluate_fn: callable | None = None,
    ):
        self.distance_matrix = distance_matrix
        self._n = len(distance_matrix)
        self._evaluate_fn = evaluate_fn

    def cost(self, solution: list[int], i: int) -> float:
        """Cost contribution of element at index i."""
        n = len(solution)
        if n < 2:
            return 0.0

        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        prev_node = solution[prev_idx]
        curr_node = solution[i]
        next_node = solution[next_idx]

        return self.distance_matrix[prev_node][curr_node] + self.distance_matrix[curr_node][next_node]

    def delta(self, solution: list[int], move_type: str, i: int, j: int) -> float:
        """Cost change if move applied."""
        n = len(solution)
        if n < 2 or i == j:
            return 0.0

        if move_type == "swap":
            # Estimate swap delta
            old_cost = self.cost(solution, i) + self.cost(solution, j)
            # Create temporary swap
            temp = solution[:]
            temp[i], temp[j] = temp[j], temp[i]
            new_cost = self.cost(temp, i) + self.cost(temp, j)
            return new_cost - old_cost

        elif move_type == "reverse":
            # 2-opt reverse
            if i > j:
                i, j = j, i
            # Calculate delta for reversing segment [i, j]
            if j - i < 1:
                return 0.0

            a, b = solution[i - 1] if i > 0 else solution[-1], solution[i]
            c, d = solution[j], solution[(j + 1) % n]

            old = self.distance_matrix[a][b] + self.distance_matrix[c][d]
            new = self.distance_matrix[a][c] + self.distance_matrix[b][d]
            return new - old

        elif move_type == "insert":
            # Move element from i to after j
            return 0.0  # Simplified

        return 0.0

    def neighbors(self, solution: list[int], i: int, k: int) -> list[int]:
        """Returns k indices most related to index i (by distance)."""
        if i >= len(solution):
            return []

        node = solution[i]
        n = len(solution)

        # Find k nearest nodes by distance
        distances = []
        for j in range(n):
            if j != i:
                other_node = solution[j]
                dist = self.distance_matrix[node][other_node]
                distances.append((dist, j))

        distances.sort()
        return [idx for _, idx in distances[:k]]

    def evaluate(self, solution: list[int]) -> float:
        """Total solution cost (uses injected evaluate_fn if available)."""
        if self._evaluate_fn is not None:
            return self._evaluate_fn(solution)

        # Fallback: TSP tour cost
        if len(solution) < 2:
            return 0.0

        total = 0.0
        for i in range(len(solution)):
            j = (i + 1) % len(solution)
            total += self.distance_matrix[solution[i]][solution[j]]
        return total

    def valid(self, solution: list[int]) -> bool:
        """Check feasibility (permutation validity)."""
        if not solution:
            return False
        n = len(solution)
        return len(set(solution)) == n and all(0 <= x < self._n for x in solution)


def _create_operator_ctx(
    source_context: "SourceContext",
    evaluate_fn: callable | None = None,
) -> TransferContext:
    """Create operator context from SourceContext."""
    return TransferContext(source_context.distance_matrix, evaluate_fn)

# Type variables for generic typing
TargetSolution = TypeVar("TargetSolution")
TargetInstance = TypeVar("TargetInstance")
SourceSolution = TypeVar("SourceSolution")


@dataclass
class AdapterConfig:
    """Configuration for domain adapter."""

    # Source domain info
    source_domain: str = "tsp"

    # How to handle capacity violations in VRP
    repair_violations: bool = True

    # Split strategy for giant tour → routes
    split_strategy: str = "greedy"  # "greedy", "optimal", "random"

    # Extra parameters
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceContext:
    """Context passed to source domain operators.

    Contiene lo necesario para que operadores TSP funcionen
    sin conocer el dominio target.
    """

    # Distance matrix in source representation
    distance_matrix: list[list[float]]

    # Mapping from source indices to target elements
    index_to_element: dict[int, Any] = field(default_factory=dict)
    element_to_index: dict[Any, int] = field(default_factory=dict)

    # Extra context (demands, capacity, etc for VRP)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptedOperator:
    """An operator adapted for a target domain.

    Wraps a source domain operator with adapter logic.
    """

    operator_id: str
    original_id: str
    role: AbstractRole
    source_domain: str
    target_domain: str
    weight: float = 1.0
    description: str = ""

    # The adapted function
    adapted_fn: callable | None = None

    # Original code (for debugging/analysis)
    original_code: str = ""


class DomainAdapter(ABC, Generic[TargetSolution, TargetInstance]):
    """Abstract base class for domain adaptation.

    Subclasses implement the specific mapping between domains.

    Principios de diseño:
    1. Encapsulación: El adaptador oculta la representación interna
    2. Bidireccionalidad: to_source y from_source son inversas
    3. Contexto: SourceContext lleva info necesaria para operadores fuente
    """

    def __init__(
        self,
        target_instance: TargetInstance,
        config: AdapterConfig | None = None,
    ):
        """Initialize adapter.

        Args:
            target_instance: Instance from target domain (e.g., VRPInstance)
            config: Optional adapter configuration
        """
        self.target_instance = target_instance
        self.config = config or AdapterConfig()
        self._context: SourceContext | None = None

    @property
    @abstractmethod
    def source_domain(self) -> str:
        """Domain name of the source (e.g., 'tsp')."""
        ...

    @property
    @abstractmethod
    def target_domain(self) -> str:
        """Domain name of the target (e.g., 'vrp')."""
        ...

    @abstractmethod
    def to_source_repr(
        self,
        target_solution: TargetSolution,
    ) -> tuple[SourceSolution, SourceContext]:
        """Convert target solution to source representation.

        Args:
            target_solution: Solution in target domain format

        Returns:
            Tuple of (source_solution, context)
            - source_solution: Solution in source domain format
            - context: SourceContext with distance matrix and mappings
        """
        ...

    @abstractmethod
    def from_source_repr(
        self,
        source_solution: SourceSolution,
        context: SourceContext | None = None,
    ) -> TargetSolution:
        """Convert source solution back to target representation.

        Args:
            source_solution: Solution in source domain format
            context: SourceContext (uses cached if not provided)

        Returns:
            Solution in target domain format
        """
        ...

    @abstractmethod
    def create_source_context(self) -> SourceContext:
        """Create context for source domain operators.

        Called once per instance to build the mapping.

        Returns:
            SourceContext with distance matrix and element mappings
        """
        ...

    @abstractmethod
    def validate_result(
        self,
        result: TargetSolution,
    ) -> bool:
        """Validate that result is feasible in target domain.

        Args:
            result: Solution in target domain format

        Returns:
            True if solution is valid/feasible
        """
        ...

    def get_context(self) -> SourceContext:
        """Get or create source context."""
        if self._context is None:
            self._context = self.create_source_context()
        return self._context

    def get_evaluate_fn(self) -> callable:
        """Get evaluation function for target domain.

        Returns a function that takes a permutation (list[int])
        and returns the cost in the target domain.

        Subclasses should override this to provide domain-specific evaluation.
        Default returns None (falls back to TSP tour cost).
        """
        return None

    def adapt_operator(
        self,
        operator_id: str,
        operator_fn: callable,
        role: AbstractRole,
        original_code: str = "",
    ) -> AdaptedOperator:
        """Adapt a source domain operator for target domain.

        Creates a wrapper function that:
        1. Converts target solution → source representation
        2. Applies the source operator
        3. Converts result → target representation
        4. Validates and repairs if needed

        Args:
            operator_id: Unique ID for the adapted operator
            operator_fn: The source domain operator function
            role: Abstract role this operator implements
            original_code: Original source code (for debugging)

        Returns:
            AdaptedOperator ready for use in target domain
        """
        context = self.get_context()

        def adapted_fn(
            target_solution: TargetSolution,
            target_instance: TargetInstance = None,
        ) -> TargetSolution:
            """Wrapped operator for target domain."""
            # Use instance from adapter if not provided
            instance = target_instance or self.target_instance

            # Convert to source representation
            source_sol, ctx = self.to_source_repr(target_solution)

            # Apply source operator
            # synthesized operators use ctx interface (ctx.evaluate, etc.)
            # Create a compatible ctx object from SourceContext
            # with evaluate_fn from target domain
            operator_ctx = _create_operator_ctx(ctx, self.get_evaluate_fn())

            try:
                improved = operator_fn(source_sol, operator_ctx)
            except Exception:
                # If operator fails, return original
                return target_solution

            # Convert back to target representation
            result = self.from_source_repr(improved, ctx)

            # Validate and return
            if self.validate_result(result):
                return result

            # If invalid and repair enabled, try to repair
            if self.config.repair_violations:
                repaired = self._repair_solution(result)
                if self.validate_result(repaired):
                    return repaired

            # Return original if all else fails
            return target_solution

        return AdaptedOperator(
            operator_id=f"{operator_id}_{self.target_domain}",
            original_id=operator_id,
            role=role,
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            weight=1.0,
            description=f"Adapted from {self.source_domain}: {operator_id}",
            adapted_fn=adapted_fn,
            original_code=original_code,
        )

    def _repair_solution(self, solution: TargetSolution) -> TargetSolution:
        """Attempt to repair an invalid solution.

        Default implementation returns solution unchanged.
        Subclasses can override for domain-specific repair.
        """
        return solution

    def apply_operator(
        self,
        target_solution: TargetSolution,
        source_operator: callable,
    ) -> TargetSolution:
        """Apply a source operator directly to target solution.

        Convenience method for one-off operator application.
        For repeated use, use adapt_operator() to create a wrapper.

        Args:
            target_solution: Solution in target domain format
            source_operator: Operator from source domain

        Returns:
            Improved solution in target domain format
        """
        context = self.get_context()

        # Convert
        source_sol, ctx = self.to_source_repr(target_solution)

        # Create operator context compatible with synthesized interface
        operator_ctx = _create_operator_ctx(ctx)

        # Apply
        try:
            improved = source_operator(source_sol, operator_ctx)
        except Exception:
            return target_solution

        # Convert back
        result = self.from_source_repr(improved, ctx)

        # Validate
        if self.validate_result(result):
            return result

        if self.config.repair_violations:
            repaired = self._repair_solution(result)
            if self.validate_result(repaired):
                return repaired

        return target_solution
