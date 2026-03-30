"""Permutation Flow Shop Problem (PFSP) domain implementation.

PFSP es una simplificación de JSSP donde todos los jobs pasan por
las máquinas en el mismo orden (1, 2, ..., m).

Representación: Permutación de jobs [0, 1, ..., n-1]
Objetivo: Minimizar makespan (tiempo total de completar todos los jobs)
"""

import random
from pathlib import Path

from pydantic import BaseModel, Field


class PFSPInstance(BaseModel):
    """PFSP problem instance.

    Attributes:
        name: Instance identifier
        n_jobs: Number of jobs
        n_machines: Number of machines
        processing_times: processing_times[job][machine] = time
        optimal_makespan: Known optimal (if available)
    """

    name: str
    n_jobs: int = Field(gt=0)
    n_machines: int = Field(gt=0)
    processing_times: list[list[int]]  # [job][machine]
    optimal_makespan: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class PFSPSolution(BaseModel):
    """PFSP solution.

    Attributes:
        sequence: Permutation of jobs [0, 1, ..., n-1]
        makespan: Total completion time
    """

    sequence: list[int]
    makespan: int = 0

    @property
    def cost(self) -> float:
        return float(self.makespan)


class PFSPDomain:
    """PFSP domain implementation."""

    @property
    def name(self) -> str:
        return "pfsp"

    def load_instance(self, path: Path) -> PFSPInstance:
        """Load PFSP instance from Taillard format.

        Format:
            n_jobs n_machines
            processing_times (one row per job)
        """
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        first_line = lines[0].split()
        n_jobs = int(first_line[0])
        n_machines = int(first_line[1])

        processing_times = []
        for i in range(1, n_jobs + 1):
            times = list(map(int, lines[i].split()))
            processing_times.append(times)

        # Try to get optimal from filename or content
        optimal = None
        for line in lines:
            if "optimal" in line.lower():
                try:
                    optimal = int(line.split()[-1])
                except ValueError:
                    pass

        return PFSPInstance(
            name=path.stem,
            n_jobs=n_jobs,
            n_machines=n_machines,
            processing_times=processing_times,
            optimal_makespan=optimal,
        )

    def evaluate_solution(self, solution: PFSPSolution, instance: PFSPInstance) -> float:
        """Compute makespan for a job sequence.

        In PFSP, jobs pass through machines 0, 1, ..., m-1 in order.
        """
        makespan = self._compute_makespan(solution.sequence, instance)
        return float(makespan)

    def _compute_makespan(self, sequence: list[int], instance: PFSPInstance) -> int:
        """Compute makespan for a job sequence.

        Args:
            sequence: Order of jobs
            instance: PFSP instance

        Returns:
            Makespan value
        """
        n_jobs = len(sequence)
        n_machines = instance.n_machines

        # completion[j][m] = completion time of job j on machine m
        # We only need to track completion time of last job on each machine
        # and completion time of current job on previous machine

        machine_completion = [0] * n_machines

        for job in sequence:
            # Job goes through machines 0, 1, ..., m-1 in order
            job_completion = 0
            for m in range(n_machines):
                # Start when both machine is free AND previous operation of this job is done
                start = max(machine_completion[m], job_completion)
                proc_time = instance.processing_times[job][m]
                end = start + proc_time

                machine_completion[m] = end
                job_completion = end

        return max(machine_completion)

    def validate_solution(self, solution: PFSPSolution, instance: PFSPInstance) -> bool:
        """Check if solution is a valid permutation."""
        seq = solution.sequence
        if len(seq) != instance.n_jobs:
            return False
        if set(seq) != set(range(instance.n_jobs)):
            return False
        return True

    def random_solution(self, instance: PFSPInstance) -> PFSPSolution:
        """Generate random permutation."""
        sequence = list(range(instance.n_jobs))
        random.shuffle(sequence)
        makespan = self._compute_makespan(sequence, instance)
        return PFSPSolution(sequence=sequence, makespan=makespan)

    def neh_solution(self, instance: PFSPInstance) -> PFSPSolution:
        """NEH heuristic - classic PFSP constructive heuristic.

        1. Sort jobs by decreasing total processing time
        2. Insert jobs one by one in best position
        """
        n_jobs = instance.n_jobs

        # Calculate total processing time for each job
        total_times = []
        for j in range(n_jobs):
            total = sum(instance.processing_times[j])
            total_times.append((total, j))

        # Sort by decreasing total time
        total_times.sort(reverse=True)
        sorted_jobs = [j for _, j in total_times]

        # Build sequence by insertion
        sequence = [sorted_jobs[0]]

        for i in range(1, n_jobs):
            job = sorted_jobs[i]
            best_pos = 0
            best_makespan = float("inf")

            # Try inserting at each position
            for pos in range(len(sequence) + 1):
                test_seq = sequence[:pos] + [job] + sequence[pos:]
                makespan = self._compute_makespan(test_seq, instance)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_pos = pos

            sequence = sequence[:best_pos] + [job] + sequence[best_pos:]

        makespan = self._compute_makespan(sequence, instance)
        return PFSPSolution(sequence=sequence, makespan=makespan)

    def spt_solution(self, instance: PFSPInstance) -> PFSPSolution:
        """Shortest Processing Time - sort by total processing time."""
        total_times = []
        for j in range(instance.n_jobs):
            total = sum(instance.processing_times[j])
            total_times.append((total, j))

        total_times.sort()  # Ascending (shortest first)
        sequence = [j for _, j in total_times]
        makespan = self._compute_makespan(sequence, instance)
        return PFSPSolution(sequence=sequence, makespan=makespan)


def create_sample_pfsp_instance(
    n_jobs: int = 20,
    n_machines: int = 5,
    seed: int = 42,
) -> PFSPInstance:
    """Create a sample PFSP instance for testing."""
    random.seed(seed)

    processing_times = []
    for _ in range(n_jobs):
        times = [random.randint(1, 99) for _ in range(n_machines)]
        processing_times.append(times)

    return PFSPInstance(
        name=f"sample_{n_jobs}x{n_machines}",
        n_jobs=n_jobs,
        n_machines=n_machines,
        processing_times=processing_times,
    )
