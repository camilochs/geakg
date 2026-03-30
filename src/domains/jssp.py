"""Job Shop Scheduling Problem (JSSP) domain implementation."""

import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from src.domains.base import OptimizationDomain, ProblemFeatures


class JSSPInstance(BaseModel):
    """JSSP problem instance.

    A JSSP instance consists of:
    - n jobs, each with m operations
    - Each operation has a machine assignment and processing time
    - Goal: minimize makespan (total completion time)
    """

    name: str
    n_jobs: int = Field(gt=0)
    n_machines: int = Field(gt=0)
    # processing_times[job][operation] = processing time
    processing_times: list[list[int]]
    # machine_assignments[job][operation] = machine id (0-indexed)
    machine_assignments: list[list[int]]
    optimal_makespan: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def dimension(self) -> int:
        """Total number of operations."""
        return self.n_jobs * self.n_machines


class JSSPSolution(BaseModel):
    """JSSP solution (schedule).

    The schedule is represented as a permutation of operations,
    where each operation is identified by (job, operation_index).
    """

    # Schedule as list of (job_id, operation_idx) tuples
    schedule: list[tuple[int, int]]
    makespan: int = Field(ge=0, default=0)
    is_valid: bool = True

    @property
    def cost(self) -> float:
        """Alias for makespan as cost."""
        return float(self.makespan)


class JSSPFeatures(ProblemFeatures):
    """Features extracted from JSSP instance."""

    dimension: int
    n_jobs: int
    n_machines: int
    avg_processing_time: float
    std_processing_time: float
    max_processing_time: int
    machine_load_balance: float  # How evenly distributed work is across machines

    @classmethod
    def from_instance(cls, instance: JSSPInstance) -> "JSSPFeatures":
        """Extract features from JSSP instance."""
        all_times = [t for job_times in instance.processing_times for t in job_times]

        avg_time = sum(all_times) / len(all_times) if all_times else 0
        variance = sum((t - avg_time) ** 2 for t in all_times) / len(all_times) if all_times else 0
        std_time = variance ** 0.5

        # Calculate machine load balance
        machine_loads = [0] * instance.n_machines
        for job_idx, job_machines in enumerate(instance.machine_assignments):
            for op_idx, machine in enumerate(job_machines):
                machine_loads[machine] += instance.processing_times[job_idx][op_idx]

        avg_load = sum(machine_loads) / instance.n_machines
        load_variance = sum((l - avg_load) ** 2 for l in machine_loads) / instance.n_machines
        load_balance = 1 - (load_variance ** 0.5 / avg_load if avg_load > 0 else 0)

        return cls(
            dimension=instance.dimension,
            n_jobs=instance.n_jobs,
            n_machines=instance.n_machines,
            avg_processing_time=avg_time,
            std_processing_time=std_time,
            max_processing_time=max(all_times) if all_times else 0,
            machine_load_balance=max(0, min(1, load_balance)),
        )


class JSSPDomain(OptimizationDomain[JSSPInstance, JSSPSolution]):
    """JSSP domain implementation."""

    @property
    def name(self) -> str:
        return "jssp"

    def load_instance(self, path: Path) -> JSSPInstance:
        """Load JSSP instance from standard format.

        Supports both Taillard format and OR-Library format.

        Args:
            path: Path to instance file

        Returns:
            Loaded JSSP instance
        """
        with open(path) as f:
            content = f.read()

        return self._parse_instance(content, path.stem)

    def _parse_instance(self, content: str, name: str) -> JSSPInstance:
        """Parse JSSP instance content.

        Args:
            content: File content
            name: Instance name

        Returns:
            Parsed instance
        """
        lines = [l.strip() for l in content.strip().split("\n") if l.strip() and not l.startswith("#")]

        # First line: n_jobs n_machines
        first_line = lines[0].split()
        n_jobs = int(first_line[0])
        n_machines = int(first_line[1])

        processing_times = []
        machine_assignments = []

        # Parse job lines
        for job_idx in range(n_jobs):
            line = lines[1 + job_idx].split()
            times = []
            machines = []

            # Format: machine1 time1 machine2 time2 ...
            for i in range(0, len(line), 2):
                machines.append(int(line[i]))
                times.append(int(line[i + 1]))

            machine_assignments.append(machines)
            processing_times.append(times)

        # Try to extract optimal from filename or content
        optimal = None
        opt_match = re.search(r"optimal[:\s]+(\d+)", content.lower())
        if opt_match:
            optimal = int(opt_match.group(1))

        return JSSPInstance(
            name=name,
            n_jobs=n_jobs,
            n_machines=n_machines,
            processing_times=processing_times,
            machine_assignments=machine_assignments,
            optimal_makespan=optimal,
        )

    def evaluate_solution(self, solution: JSSPSolution, instance: JSSPInstance) -> float:
        """Compute makespan of a schedule.

        Args:
            solution: JSSP solution (schedule)
            instance: JSSP instance

        Returns:
            Makespan (lower is better)
        """
        makespan = self._compute_makespan(solution.schedule, instance)
        return float(makespan)

    def _compute_makespan(
        self, schedule: list[tuple[int, int]], instance: JSSPInstance
    ) -> int:
        """Compute makespan for a schedule.

        Args:
            schedule: List of (job_id, operation_idx) tuples
            instance: JSSP instance

        Returns:
            Makespan value
        """
        # Track completion times
        job_completion = [0] * instance.n_jobs  # When each job's last op finished
        machine_completion = [0] * instance.n_machines  # When each machine is free

        for job_id, op_idx in schedule:
            machine = instance.machine_assignments[job_id][op_idx]
            processing_time = instance.processing_times[job_id][op_idx]

            # Operation can start when both job's previous op and machine are free
            start_time = max(job_completion[job_id], machine_completion[machine])
            end_time = start_time + processing_time

            job_completion[job_id] = end_time
            machine_completion[machine] = end_time

        return max(machine_completion)

    def validate_solution(self, solution: JSSPSolution, instance: JSSPInstance) -> bool:
        """Check if schedule is valid.

        Args:
            solution: JSSP solution
            instance: JSSP instance

        Returns:
            True if valid schedule
        """
        schedule = solution.schedule

        # Check all operations are present
        expected = set()
        for job in range(instance.n_jobs):
            for op in range(instance.n_machines):
                expected.add((job, op))

        actual = set(schedule)
        if actual != expected:
            return False

        # Check precedence constraints (operations of same job in order)
        job_op_counts = [0] * instance.n_jobs
        for job_id, op_idx in schedule:
            if op_idx != job_op_counts[job_id]:
                return False
            job_op_counts[job_id] += 1

        return True

    def get_features(self, instance: JSSPInstance) -> JSSPFeatures:
        """Extract features from JSSP instance.

        Args:
            instance: JSSP instance

        Returns:
            Extracted features
        """
        return JSSPFeatures.from_instance(instance)

    def random_solution(self, instance: JSSPInstance) -> JSSPSolution:
        """Generate a random valid schedule.

        Uses random priority dispatch.

        Args:
            instance: JSSP instance

        Returns:
            Random valid schedule
        """
        # Track next operation for each job
        next_op = [0] * instance.n_jobs
        schedule = []

        total_ops = instance.n_jobs * instance.n_machines

        while len(schedule) < total_ops:
            # Get jobs with remaining operations
            available = [j for j in range(instance.n_jobs) if next_op[j] < instance.n_machines]

            # Random selection
            job = random.choice(available)
            op = next_op[job]

            schedule.append((job, op))
            next_op[job] += 1

        solution = JSSPSolution(schedule=schedule)
        solution.makespan = self._compute_makespan(schedule, instance)
        return solution

    def priority_dispatch_solution(
        self,
        instance: JSSPInstance,
        priority: str = "spt",  # spt, lpt, random
    ) -> JSSPSolution:
        """Generate solution using priority dispatching rules.

        Args:
            instance: JSSP instance
            priority: Priority rule (spt=shortest processing time, lpt=longest)

        Returns:
            Dispatched schedule
        """
        next_op = [0] * instance.n_jobs
        schedule = []
        total_ops = instance.n_jobs * instance.n_machines

        while len(schedule) < total_ops:
            available = [j for j in range(instance.n_jobs) if next_op[j] < instance.n_machines]

            if priority == "spt":
                # Shortest processing time first
                job = min(
                    available,
                    key=lambda j: instance.processing_times[j][next_op[j]],
                )
            elif priority == "lpt":
                # Longest processing time first
                job = max(
                    available,
                    key=lambda j: instance.processing_times[j][next_op[j]],
                )
            else:
                job = random.choice(available)

            op = next_op[job]
            schedule.append((job, op))
            next_op[job] += 1

        solution = JSSPSolution(schedule=schedule)
        solution.makespan = self._compute_makespan(schedule, instance)
        return solution

    def simulated_annealing_solution(
        self,
        instance: JSSPInstance,
        time_limit: float = 60.0,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        initial_solution: JSSPSolution | None = None,
    ) -> JSSPSolution:
        """Solve JSSP using Simulated Annealing.

        More effective than basic Tabu Search for JSSP.

        Args:
            instance: JSSP instance
            time_limit: Time limit in seconds
            initial_temp: Initial temperature
            cooling_rate: Temperature cooling rate
            initial_solution: Optional initial solution (uses SPT if None)

        Returns:
            Best solution found
        """
        import time
        import random
        import math

        start_time = time.time()

        # Initialize
        if initial_solution is None:
            current = self.priority_dispatch_solution(instance, "spt")
        else:
            current = initial_solution

        current_makespan = self._compute_makespan(current.schedule, instance)
        best = current
        best_makespan = current_makespan

        temp = initial_temp
        iterations = 0

        while time.time() - start_time < time_limit:
            iterations += 1

            # Generate neighbor by swapping two random valid positions
            schedule = current.schedule[:]

            # Try multiple times to find valid swap
            for _ in range(10):
                i = random.randint(0, len(schedule) - 2)
                j = i + 1

                # Swap
                schedule[i], schedule[j] = schedule[j], schedule[i]

                # Check validity
                if self._is_valid_schedule(schedule, instance):
                    break
                else:
                    # Undo if invalid
                    schedule[i], schedule[j] = schedule[j], schedule[i]

            new_makespan = self._compute_makespan(schedule, instance)
            delta = new_makespan - current_makespan

            # Accept if better or with probability based on temperature
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = JSSPSolution(schedule=schedule, makespan=new_makespan)
                current_makespan = new_makespan

                # Update best
                if new_makespan < best_makespan:
                    best = current
                    best_makespan = new_makespan

            # Cool down
            if iterations % 100 == 0:
                temp *= cooling_rate

        return best

    def tabu_search_solution(
        self,
        instance: JSSPInstance,
        time_limit: float = 60.0,
        tabu_tenure: int = 10,
        initial_solution: JSSPSolution | None = None,
    ) -> JSSPSolution:
        """Solve JSSP using Tabu Search with N1 neighborhood (swap operations).

        Simple but effective implementation using swap moves.

        Args:
            instance: JSSP instance
            time_limit: Time limit in seconds
            tabu_tenure: Tabu list tenure
            initial_solution: Optional initial solution (uses SPT if None)

        Returns:
            Best solution found
        """
        import time

        start_time = time.time()

        # Initialize with SPT if no initial solution
        if initial_solution is None:
            current = self.priority_dispatch_solution(instance, "spt")
        else:
            current = initial_solution

        current_makespan = self._compute_makespan(current.schedule, instance)
        best = current
        best_makespan = current_makespan

        # Tabu list: stores forbidden swap positions as (i, j, iteration)
        tabu_list: list[tuple[int, int, int]] = []
        iterations = 0
        no_improvement = 0

        while time.time() - start_time < time_limit and no_improvement < 50:
            iterations += 1

            # Generate neighborhood: all valid swaps
            neighbors = []
            schedule = current.schedule

            for i in range(len(schedule) - 1):
                # Try swapping i with i+1
                new_schedule = schedule[:]
                new_schedule[i], new_schedule[i + 1] = new_schedule[i + 1], new_schedule[i]

                # Check validity
                if self._is_valid_schedule(new_schedule, instance):
                    new_makespan = self._compute_makespan(new_schedule, instance)
                    neighbors.append((new_schedule, new_makespan, i, i + 1))

            if not neighbors:
                break

            # Select best non-tabu move (aspiration: accept if better than best ever)
            best_neighbor = None
            best_neighbor_makespan = float("inf")
            best_neighbor_swap = None

            for schedule_n, makespan, pos1, pos2 in neighbors:
                # Check if move is tabu
                is_tabu = any(
                    (pos1 == t_pos1 and pos2 == t_pos2) or (pos2 == t_pos1 and pos1 == t_pos2)
                    for t_pos1, t_pos2, _ in tabu_list
                )

                # Aspiration criterion: accept if better than best ever
                if makespan < best_makespan:
                    best_neighbor = schedule_n
                    best_neighbor_makespan = makespan
                    best_neighbor_swap = (pos1, pos2)

                # Accept best non-tabu
                elif not is_tabu and makespan < best_neighbor_makespan:
                    best_neighbor = schedule_n
                    best_neighbor_makespan = makespan
                    best_neighbor_swap = (pos1, pos2)

            if best_neighbor is None:
                break

            # Update current solution
            current = JSSPSolution(schedule=best_neighbor, makespan=best_neighbor_makespan)
            current_makespan = best_neighbor_makespan

            # Update best if improved
            if current_makespan < best_makespan:
                best = current
                best_makespan = current_makespan
                no_improvement = 0
            else:
                no_improvement += 1

            # Add move to tabu list
            tabu_list.append((best_neighbor_swap[0], best_neighbor_swap[1], iterations))

            # Remove expired tabu entries
            tabu_list = [(p1, p2, it) for p1, p2, it in tabu_list if iterations - it < tabu_tenure]

        return best

    def _find_critical_path(
        self, schedule: list[tuple[int, int]], instance: JSSPInstance
    ) -> list[tuple[int, int]]:
        """Find critical path in schedule (operations on longest path to makespan).

        Args:
            schedule: Current schedule
            instance: JSSP instance

        Returns:
            List of operations on critical path
        """
        # Compute start and end times for all operations
        job_completion = [0] * instance.n_jobs
        machine_completion = [0] * instance.n_machines
        op_times = {}  # (job, op) -> (start, end)

        for job_id, op_idx in schedule:
            machine = instance.machine_assignments[job_id][op_idx]
            processing_time = instance.processing_times[job_id][op_idx]

            start_time = max(job_completion[job_id], machine_completion[machine])
            end_time = start_time + processing_time

            op_times[(job_id, op_idx)] = (start_time, end_time)

            job_completion[job_id] = end_time
            machine_completion[machine] = end_time

        makespan = max(machine_completion)

        # Backtrack from operation that ends at makespan
        critical = []
        # Find last operation (the one ending at makespan)
        for job_id, op_idx in reversed(schedule):
            start, end = op_times[(job_id, op_idx)]
            if end == makespan:
                critical.append((job_id, op_idx))
                current_time = start
                current_job = job_id
                current_machine = instance.machine_assignments[job_id][op_idx]

                # Trace back through critical path
                while current_time > 0:
                    # Find predecessor on critical path (job or machine constraint)
                    found = False

                    # Check job predecessor
                    if op_idx > 0:
                        pred_op = (current_job, op_idx - 1)
                        if pred_op in op_times:
                            pred_start, pred_end = op_times[pred_op]
                            if pred_end == current_time:
                                critical.append(pred_op)
                                current_time = pred_start
                                current_job = pred_op[0]
                                op_idx = pred_op[1]
                                current_machine = instance.machine_assignments[current_job][op_idx]
                                found = True

                    # Check machine predecessor if not found
                    if not found:
                        for j, o in reversed(schedule):
                            if (j, o) in op_times:
                                s, e = op_times[(j, o)]
                                if instance.machine_assignments[j][o] == current_machine and e == current_time:
                                    critical.append((j, o))
                                    current_time = s
                                    current_job = j
                                    op_idx = o
                                    found = True
                                    break

                    if not found:
                        break

                break

        return list(reversed(critical))

    def _is_valid_schedule(
        self, schedule: list[tuple[int, int]], instance: JSSPInstance
    ) -> bool:
        """Quick validity check for schedule (precedence constraints).

        Args:
            schedule: Schedule to check
            instance: JSSP instance

        Returns:
            True if schedule respects precedence
        """
        job_op_counts = [0] * instance.n_jobs
        for job_id, op_idx in schedule:
            if op_idx != job_op_counts[job_id]:
                return False
            job_op_counts[job_id] += 1
        return True


def create_sample_jssp_instance(
    n_jobs: int = 6,
    n_machines: int = 6,
    seed: int = 42,
) -> JSSPInstance:
    """Create a sample JSSP instance for testing.

    Args:
        n_jobs: Number of jobs
        n_machines: Number of machines
        seed: Random seed

    Returns:
        Sample JSSP instance
    """
    random.seed(seed)

    processing_times = []
    machine_assignments = []

    for _ in range(n_jobs):
        # Random processing times between 1 and 99
        times = [random.randint(1, 99) for _ in range(n_machines)]
        # Random machine order (each machine exactly once)
        machines = list(range(n_machines))
        random.shuffle(machines)

        processing_times.append(times)
        machine_assignments.append(machines)

    return JSSPInstance(
        name=f"sample_{n_jobs}x{n_machines}",
        n_jobs=n_jobs,
        n_machines=n_machines,
        processing_times=processing_times,
        machine_assignments=machine_assignments,
    )
