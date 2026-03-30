"""PFSP Adapter: Transfer TSP operators to PFSP domain.

PFSP es el adaptador más simple porque:
- TSP: permutación [0, 1, ..., n-1]
- PFSP: permutación [0, 1, ..., n-1]

Son exactamente la misma representación. Solo necesitamos
construir una matriz de "distancias" para los operadores TSP.

La "distancia" se basa en diferencia de tiempos de procesamiento.
"""

from dataclasses import dataclass
from typing import Any

from src.geakg.transfer.adapter import DomainAdapter, AdapterConfig, SourceContext


@dataclass
class PFSPSequence:
    """PFSP solution representation."""

    sequence: list[int]  # Permutation of jobs [0, 1, ..., n-1]
    makespan: int = 0

    @property
    def cost(self) -> float:
        return float(self.makespan)


class PFSPAdapter(DomainAdapter[PFSPSequence, Any]):
    """Adapter for transferring TSP operators to PFSP.

    Mapping is trivial:
    - TSP tour [0, 1, ..., n-1] = PFSP sequence [0, 1, ..., n-1]

    Distance matrix is constructed from processing times:
    - d[i][j] = |total_time[i] - total_time[j]|
    """

    def __init__(
        self,
        pfsp_instance: Any,
        config: AdapterConfig | None = None,
    ):
        super().__init__(pfsp_instance, config)
        self._setup_mappings()

    def _setup_mappings(self) -> None:
        """Setup job index mappings (trivial for PFSP)."""
        instance = self.target_instance
        self.n_jobs = instance.n_jobs

        # Calculate total processing time per job
        self.total_times = []
        for j in range(instance.n_jobs):
            total = sum(instance.processing_times[j])
            self.total_times.append(total)

    @property
    def source_domain(self) -> str:
        return "tsp"

    @property
    def target_domain(self) -> str:
        return "pfsp"

    def create_source_context(self) -> SourceContext:
        """Create TSP context from PFSP instance.

        Distance matrix based on processing time differences.
        Jobs with similar total times are "closer".
        """
        n = self.n_jobs

        # Distance = difference in total processing times
        tsp_dm = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    tsp_dm[i][j] = 1.0 + abs(self.total_times[i] - self.total_times[j])

        extra = {
            "n_jobs": self.n_jobs,
            "n_machines": self.target_instance.n_machines,
            "processing_times": self.target_instance.processing_times,
            "total_times": self.total_times,
        }

        return SourceContext(
            distance_matrix=tsp_dm,
            index_to_element={i: i for i in range(n)},
            element_to_index={i: i for i in range(n)},
            extra=extra,
        )

    def to_source_repr(
        self,
        pfsp_solution: PFSPSequence | list[int],
    ) -> tuple[list[int], SourceContext]:
        """Convert PFSP sequence to TSP tour.

        Trivial: they're the same representation!
        """
        if isinstance(pfsp_solution, PFSPSequence):
            sequence = pfsp_solution.sequence
        else:
            sequence = pfsp_solution

        # Direct mapping
        tsp_tour = list(sequence)
        context = self.get_context()
        return tsp_tour, context

    def from_source_repr(
        self,
        tsp_tour: list[int],
        context: SourceContext | None = None,
    ) -> PFSPSequence:
        """Convert TSP tour back to PFSP sequence.

        Trivial: they're the same representation!
        """
        sequence = list(tsp_tour)
        makespan = self._compute_makespan(sequence)
        return PFSPSequence(sequence=sequence, makespan=makespan)

    def _compute_makespan(self, sequence: list[int]) -> int:
        """Compute makespan for a job sequence."""
        instance = self.target_instance
        n_machines = instance.n_machines

        machine_completion = [0] * n_machines

        for job in sequence:
            job_completion = 0
            for m in range(n_machines):
                start = max(machine_completion[m], job_completion)
                proc_time = instance.processing_times[job][m]
                end = start + proc_time
                machine_completion[m] = end
                job_completion = end

        return max(machine_completion)

    def validate_result(self, result: PFSPSequence) -> bool:
        """Validate PFSP solution is a valid permutation."""
        seq = result.sequence
        n = self.n_jobs
        if len(seq) != n:
            return False
        if set(seq) != set(range(n)):
            return False
        return True

    def _repair_solution(self, solution: PFSPSequence) -> PFSPSequence:
        """Repair invalid permutation.

        Simple repair: ensure all jobs appear exactly once.
        """
        sequence = solution.sequence
        n = self.n_jobs

        seen = set()
        repaired = []
        for job in sequence:
            if 0 <= job < n and job not in seen:
                repaired.append(job)
                seen.add(job)

        # Add missing jobs
        for job in range(n):
            if job not in seen:
                repaired.append(job)

        makespan = self._compute_makespan(repaired)
        return PFSPSequence(sequence=repaired, makespan=makespan)


def create_pfsp_adapter(pfsp_instance: Any) -> PFSPAdapter:
    """Factory function to create PFSP adapter."""
    config = AdapterConfig(source_domain="tsp", repair_violations=True)
    return PFSPAdapter(pfsp_instance, config)
