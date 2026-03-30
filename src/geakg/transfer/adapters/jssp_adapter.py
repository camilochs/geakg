"""JSSP Adapter: Transfer TSP operators to JSSP domain.

Implementa adaptación TSP→JSSP usando representación de permutación:
- JSSP schedule: [(job, op), (job, op), ...] con restricciones de precedencia
- TSP tour: [0, 1, 2, ...] permutación simple

El adaptador:
1. Convierte JSSP schedule a permutación de índices
2. Crea matriz de "distancias" basada en tiempos de procesamiento
3. Aplica operadores TSP a la permutación
4. Reconvierte a schedule respetando precedencia de operaciones

Diferencia clave con VRP:
- VRP: restricción de capacidad (se maneja con split)
- JSSP: restricción de precedencia (operaciones del mismo job en orden)

Ejemplo:
    from src.domains.jssp import JSSPInstance
    from src.geakg.transfer import JSSPAdapter

    adapter = JSSPAdapter(jssp_instance)
    improved_schedule = adapter.apply_operator(current_schedule, tsp_operator)
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


@dataclass
class JSSPSchedule:
    """JSSP solution representation."""

    schedule: list[tuple[int, int]]  # List of (job_id, operation_idx)
    makespan: int = 0

    @property
    def cost(self) -> float:
        """Alias for makespan."""
        return float(self.makespan)


class JSSPAdapter(DomainAdapter[JSSPSchedule, Any]):
    """Adapter for transferring TSP operators to JSSP.

    Representación:
    - JSSP: schedule = [(j0,o0), (j1,o1), ...] donde (j,o) es job j, operación o
    - TSP: permutation = [i0, i1, ...] donde i es índice en lista de operaciones

    La "distancia" en JSSP se construye como:
    - d[i][j] = abs(processing_time[i] - processing_time[j])
    - Esto hace que operaciones con tiempos similares sean "cercanas"

    Restricción de precedencia:
    - Operaciones del mismo job deben ejecutarse en orden (op 0 antes de op 1, etc.)
    - Al reconvertir, se reordena para respetar precedencia
    """

    def __init__(
        self,
        jssp_instance: Any,
        config: AdapterConfig | None = None,
    ):
        """Initialize JSSP adapter.

        Args:
            jssp_instance: JSSPInstance with processing_times, machine_assignments
            config: Optional configuration
        """
        super().__init__(jssp_instance, config)
        self._setup_mappings()

    def _setup_mappings(self) -> None:
        """Setup operation ↔ index mappings."""
        instance = self.target_instance

        # All operations as (job, op) tuples
        self.operations: list[tuple[int, int]] = []
        for job in range(instance.n_jobs):
            for op in range(instance.n_machines):
                self.operations.append((job, op))

        # Bidirectional mapping
        self.idx_to_op = {i: op for i, op in enumerate(self.operations)}
        self.op_to_idx = {op: i for i, op in enumerate(self.operations)}

        # Processing time for each operation (flattened)
        self.processing_times: list[int] = []
        for job in range(instance.n_jobs):
            for op in range(instance.n_machines):
                self.processing_times.append(instance.processing_times[job][op])

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "jssp"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from JSSP instance.

        Builds distance matrix based on processing time similarity.
        Operations with similar processing times are "closer".
        """
        n = len(self.operations)

        # Create distance matrix based on processing time difference
        # This is a heuristic: similar processing times → similar "nodes"
        tsp_dm = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Distance = difference in processing times
                    # Add base distance to avoid zero distances
                    time_diff = abs(self.processing_times[i] - self.processing_times[j])
                    tsp_dm[i][j] = 1.0 + time_diff

        # Extra info for repair/validation
        extra = {
            "n_jobs": self.target_instance.n_jobs,
            "n_machines": self.target_instance.n_machines,
            "processing_times": self.target_instance.processing_times,
            "machine_assignments": self.target_instance.machine_assignments,
        }

        return SourceContext(
            distance_matrix=tsp_dm,
            index_to_element=self.idx_to_op.copy(),
            element_to_index=self.op_to_idx.copy(),
            extra=extra,
        )

    def to_source_repr(
        self,
        jssp_schedule: JSSPSchedule | list[tuple[int, int]],
    ) -> tuple[list[int], SourceContext]:
        """Convert JSSP schedule to TSP permutation.

        JSSP schedule: [(0,0), (1,0), (0,1), (1,1), ...]
        TSP permutation: [0, 2, 1, 3, ...]  (mapped indices)

        Args:
            jssp_schedule: JSSP solution (schedule or JSSPSchedule)

        Returns:
            (tsp_permutation, context)
        """
        # Handle both JSSPSchedule and raw list
        if isinstance(jssp_schedule, JSSPSchedule):
            schedule = jssp_schedule.schedule
        else:
            schedule = jssp_schedule

        # Convert (job, op) tuples to indices
        tsp_perm = [self.op_to_idx[op] for op in schedule]

        context = self.get_context()
        return tsp_perm, context

    def from_source_repr(
        self,
        tsp_perm: list[int],
        context: SourceContext | None = None,
    ) -> JSSPSchedule:
        """Convert TSP permutation back to JSSP schedule.

        Reorders to respect precedence constraints:
        - Operations of the same job must be in order (op 0 before op 1, etc.)

        Args:
            tsp_perm: TSP permutation (indices 0 to n-1)
            context: Source context (uses cached if None)

        Returns:
            JSSPSchedule with valid precedence
        """
        # Convert indices to (job, op) tuples
        raw_schedule = [self.idx_to_op[i] for i in tsp_perm]

        # Repair precedence violations
        valid_schedule = self._repair_precedence(raw_schedule)

        # Calculate makespan
        makespan = self._calculate_makespan(valid_schedule)

        return JSSPSchedule(schedule=valid_schedule, makespan=makespan)

    def _repair_precedence(
        self,
        schedule: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Repair schedule to respect job precedence constraints.

        For each job, operations must appear in order (0, 1, 2, ...).

        Strategy:
        - Scan through schedule
        - For each operation, check if its predecessor in the same job has been scheduled
        - If not, defer it and continue

        Args:
            schedule: Potentially invalid schedule

        Returns:
            Valid schedule with precedence respected
        """
        instance = self.target_instance
        n_jobs = instance.n_jobs
        n_machines = instance.n_machines

        # Track next expected operation for each job
        next_op_for_job = [0] * n_jobs

        # Build valid schedule
        valid_schedule: list[tuple[int, int]] = []
        deferred: list[tuple[int, int]] = []

        # Process operations, deferring those that violate precedence
        remaining = list(schedule)

        while remaining or deferred:
            made_progress = False

            # Try to schedule from remaining
            new_remaining = []
            for job, op in remaining:
                if op == next_op_for_job[job]:
                    valid_schedule.append((job, op))
                    next_op_for_job[job] += 1
                    made_progress = True
                else:
                    new_remaining.append((job, op))

            remaining = new_remaining

            # Try deferred operations
            new_deferred = []
            for job, op in deferred:
                if op == next_op_for_job[job]:
                    valid_schedule.append((job, op))
                    next_op_for_job[job] += 1
                    made_progress = True
                else:
                    new_deferred.append((job, op))

            deferred = new_deferred

            # Move remaining that can't be scheduled now to deferred
            if not made_progress and remaining:
                deferred.extend(remaining)
                remaining = []

            # Safety: if no progress and we have deferred ops, force schedule
            if not made_progress and deferred:
                # Find any operation that can be scheduled
                for i, (job, op) in enumerate(deferred):
                    if op == next_op_for_job[job]:
                        valid_schedule.append((job, op))
                        next_op_for_job[job] += 1
                        deferred.pop(i)
                        made_progress = True
                        break

                # If still no progress, something is wrong - add missing ops
                if not made_progress:
                    for job in range(n_jobs):
                        while next_op_for_job[job] < n_machines:
                            op = next_op_for_job[job]
                            valid_schedule.append((job, op))
                            next_op_for_job[job] += 1
                    break

        return valid_schedule

    def _calculate_makespan(self, schedule: list[tuple[int, int]]) -> int:
        """Calculate makespan for a schedule.

        Args:
            schedule: Valid JSSP schedule

        Returns:
            Makespan value
        """
        instance = self.target_instance

        # Track completion times
        job_completion = [0] * instance.n_jobs
        machine_completion = [0] * instance.n_machines

        for job_id, op_idx in schedule:
            machine = instance.machine_assignments[job_id][op_idx]
            processing_time = instance.processing_times[job_id][op_idx]

            # Operation starts when both job's previous op and machine are free
            start_time = max(job_completion[job_id], machine_completion[machine])
            end_time = start_time + processing_time

            job_completion[job_id] = end_time
            machine_completion[machine] = end_time

        return max(machine_completion) if machine_completion else 0

    def validate_result(self, result: JSSPSchedule) -> bool:
        """Validate JSSP solution.

        Checks:
        - All operations present exactly once
        - Precedence constraints satisfied (ops of same job in order)

        Args:
            result: JSSP solution

        Returns:
            True if valid
        """
        instance = self.target_instance
        schedule = result.schedule

        # Check all operations present
        expected = set(self.operations)
        actual = set(schedule)
        if actual != expected:
            return False

        # Check precedence (operations of same job in order)
        job_op_counts = [0] * instance.n_jobs
        for job_id, op_idx in schedule:
            if op_idx != job_op_counts[job_id]:
                return False
            job_op_counts[job_id] += 1

        return True

    def _repair_solution(self, solution: JSSPSchedule) -> JSSPSchedule:
        """Repair invalid JSSP solution.

        Args:
            solution: Potentially invalid solution

        Returns:
            Repaired solution
        """
        # Already handled by _repair_precedence in from_source_repr
        repaired_schedule = self._repair_precedence(solution.schedule)
        makespan = self._calculate_makespan(repaired_schedule)
        return JSSPSchedule(schedule=repaired_schedule, makespan=makespan)


def create_jssp_adapter(
    jssp_instance: Any,
) -> JSSPAdapter:
    """Factory function to create JSSP adapter.

    Args:
        jssp_instance: JSSPInstance

    Returns:
        Configured JSSPAdapter
    """
    config = AdapterConfig(
        source_domain="tsp",
        repair_violations=True,
    )
    return JSSPAdapter(jssp_instance, config)
