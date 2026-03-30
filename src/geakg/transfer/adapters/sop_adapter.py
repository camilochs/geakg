"""SOP Adapter: Transfer TSP operators to SOP domain.

SOP (Sequential Ordering Problem) is TSP with precedence constraints.
The adaptation is straightforward since both are permutation problems.

Adaptation strategy:
- SOP path → TSP tour (direct mapping)
- Apply TSP operator
- Repair if precedences violated
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


class SOPAdapter(DomainAdapter[Any, Any]):
    """Adapter for transferring TSP operators to SOP.

    Representation:
    - SOP: path = [0, 2, 1, 3, ...] (nodes in visit order)
    - TSP: tour = [0, 2, 1, 3, ...] (same representation)

    The main difference is that SOP has precedence constraints.
    """

    def __init__(
        self,
        sop_instance: Any,
        config: AdapterConfig | None = None,
    ):
        """Initialize SOP adapter.

        Args:
            sop_instance: SOPInstance with distance_matrix and precedences
            config: Optional configuration
        """
        super().__init__(sop_instance, config)

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "sop"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from SOP instance.

        Uses the SOP distance matrix directly.
        """
        instance = self.target_instance
        n = instance.n

        # Distance matrix is already in correct format
        dm = instance.distance_matrix

        # Store precedences in extra for repair
        extra = {
            "precedences": instance.precedences,
            "n": n,
        }

        return SourceContext(
            distance_matrix=dm,
            index_to_element={i: i for i in range(n)},
            element_to_index={i: i for i in range(n)},
            extra=extra,
        )

    def to_source_repr(
        self,
        sop_solution: Any,
    ) -> tuple[list[int], SourceContext]:
        """Convert SOP path to TSP tour.

        SOP path and TSP tour have same representation.

        Args:
            sop_solution: SOPSolution with path

        Returns:
            (tsp_tour, context)
        """
        # Handle both SOPSolution objects and raw lists
        if hasattr(sop_solution, "path"):
            path = sop_solution.path
        else:
            path = sop_solution

        # SOP path is already a permutation (same as TSP tour)
        tsp_tour = list(path)

        context = self.get_context()
        return tsp_tour, context

    def from_source_repr(
        self,
        tsp_tour: list[int],
        context: SourceContext | None = None,
    ) -> Any:
        """Convert TSP tour back to SOP path.

        May need to repair precedence violations.

        Args:
            tsp_tour: TSP tour (permutation)
            context: Source context (uses cached if None)

        Returns:
            SOPSolution with valid path
        """
        from src.domains.sop import SOPSolution

        ctx = context or self.get_context()

        # Repair precedence violations if any
        repaired_path = self._repair_precedences(tsp_tour, ctx)

        # Calculate length
        length = self._calculate_length(repaired_path)

        return SOPSolution(path=repaired_path, length=length, is_valid=True)

    def _repair_precedences(
        self,
        path: list[int],
        context: SourceContext,
    ) -> list[int]:
        """Repair precedence violations in path.

        Uses greedy repair: move violating nodes to valid positions.

        Args:
            path: Potentially invalid path
            context: Source context with precedences

        Returns:
            Repaired path respecting precedences
        """
        precedences = context.extra["precedences"]

        # Build precedence graph
        must_precede = {i: set() for i in range(len(path))}
        for before, after in precedences:
            must_precede[before].add(after)

        # Check and repair violations
        position = {node: i for i, node in enumerate(path)}
        repaired = list(path)

        changed = True
        max_iterations = 100
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for before, after in precedences:
                pos_before = position[before]
                pos_after = position[after]

                # If violation (before comes after 'after')
                if pos_before >= pos_after:
                    # Move 'before' to just before 'after'
                    repaired.pop(pos_before)
                    new_pos = position[after] if pos_before > position[after] else position[after] - 1
                    repaired.insert(new_pos, before)

                    # Update positions
                    position = {node: i for i, node in enumerate(repaired)}
                    changed = True
                    break

        return repaired

    def _calculate_length(self, path: list[int]) -> float:
        """Calculate path length.

        Args:
            path: Sequence of nodes

        Returns:
            Total distance
        """
        instance = self.target_instance
        dm = instance.distance_matrix

        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            total += dm[path[i]][path[i + 1]]

        return total

    def validate_result(self, result: Any) -> bool:
        """Validate SOP solution.

        Checks:
        - All nodes visited exactly once
        - Precedence constraints satisfied

        Args:
            result: SOP solution

        Returns:
            True if valid
        """
        instance = self.target_instance

        if hasattr(result, "path"):
            path = result.path
        else:
            path = result

        # Check all nodes visited
        if len(path) != instance.n or len(set(path)) != instance.n:
            return False

        # Check precedences
        position = {node: i for i, node in enumerate(path)}
        for before, after in instance.precedences:
            if position[before] >= position[after]:
                return False

        return True

    def _repair_solution(self, solution: Any) -> Any:
        """Repair invalid SOP solution.

        Args:
            solution: Potentially invalid solution

        Returns:
            Repaired solution
        """
        # Convert to tour, repair, convert back
        tour, ctx = self.to_source_repr(solution)
        repaired_tour = self._repair_precedences(tour, ctx)
        return self.from_source_repr(repaired_tour, ctx)


def create_sop_adapter(
    sop_instance: Any,
) -> SOPAdapter:
    """Factory function to create SOP adapter.

    Args:
        sop_instance: SOPInstance

    Returns:
        Configured SOPAdapter
    """
    config = AdapterConfig(
        source_domain="tsp",
        repair_violations=True,
    )
    return SOPAdapter(sop_instance, config)
