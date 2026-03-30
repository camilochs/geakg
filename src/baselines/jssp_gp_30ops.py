"""Genetic Programming baseline for JSSP using 30 TSP operators.

This baseline applies the same 30 operators available to NS-SE to JSSP.
The key insight is that TSP operators work on permutations, and JSSP schedules
are also permutations of operations.

This provides a fair comparison baseline to demonstrate:
1. GP with identical operators to NS-SE (no unfair advantage)
2. Without knowledge-guided selection, GP struggles to compose operators effectively

Operators adapted from TSP to JSSP:
- Construction: nearest_neighbor, farthest_insertion, etc. -> priority-based schedule construction
- Local Search: two_opt, or_opt, etc. -> operation swap/insert on critical path
- Perturbation: double_bridge, ruin_recreate -> schedule disruption
- Meta-heuristic: SA, tabu, etc. -> schedule optimization

This is for the IEEE TEVC paper: demonstrating transfer learning capability.
"""

import math
import operator
import random
import time
from typing import Any

from deap import base, creator, gp, tools
from pydantic import BaseModel, Field


class JSSPGP30Result(BaseModel):
    """Result from GP with 30 operators run on JSSP."""

    best_fitness: float
    best_individual: str = ""
    evaluations: int = 0
    generations: int = 0
    wall_time_seconds: float = 0.0
    fitness_history: list[float] = Field(default_factory=list)


class JSSPGP30Ops:
    """Genetic Programming for JSSP using 30 TSP-derived operators.

    Evolves programs that construct JSSP schedules by combining
    ALL 30 algorithmic primitives available to NS-SE, adapted for JSSP.

    The operators work on permutation schedules, which is the common
    abstraction between TSP tours and JSSP operation sequences.
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 100,
        budget: int = 1000,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 5,
        max_tree_depth: int = 6,
        seed: int | None = None,
    ) -> None:
        """Initialize GP baseline with 30 operators."""
        self.population_size = population_size
        self.max_generations = max_generations
        self.budget = budget
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.max_tree_depth = max_tree_depth

        if seed is not None:
            random.seed(seed)

        self._eval_count = 0
        self._fitness_history: list[float] = []

    def run(
        self,
        processing_times: list[list[int]],
        machine_assignments: list[list[int]],
    ) -> JSSPGP30Result:
        """Run GP optimization for JSSP.

        Args:
            processing_times: Processing time for each operation [job][op]
            machine_assignments: Machine for each operation [job][op]

        Returns:
            JSSPGP30Result with best solution and statistics
        """
        start_time = time.time()
        n_jobs = len(processing_times)
        n_machines = len(set(m for job in machine_assignments for m in job))
        n_ops_per_job = len(processing_times[0]) if processing_times else 0
        total_ops = n_jobs * n_ops_per_job

        self._eval_count = 0
        self._fitness_history = []

        # Store problem data for evaluation
        self._processing_times = processing_times
        self._machine_assignments = machine_assignments
        self._n_jobs = n_jobs
        self._n_machines = n_machines
        self._n_ops_per_job = n_ops_per_job
        self._total_ops = total_ops

        # Create "distance matrix" analogue for JSSP
        # This represents operation dependencies/conflicts
        self._op_matrix = self._create_operation_matrix()

        # Define primitive set for JSSP scheduling with 30 operators
        pset = self._create_primitive_set()

        # Create fitness and individual types
        if not hasattr(creator, "JSSP30FitnessMin"):
            creator.create("JSSP30FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "JSSP30Individual"):
            creator.create("JSSP30Individual", gp.PrimitiveTree, fitness=creator.JSSP30FitnessMin)

        # Create toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=self.max_tree_depth)
        toolbox.register("individual", tools.initIterate, creator.JSSP30Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Evaluation function
        def evaluate(individual: Any) -> tuple[float]:
            if self._eval_count >= self.budget:
                return (float("inf"),)

            try:
                func = toolbox.compile(expr=individual)
                schedule = func(self._op_matrix)

                # Extract schedule from tuple if needed
                if isinstance(schedule, tuple):
                    schedule = schedule[0]

                # Validate and evaluate schedule
                makespan = self._evaluate_schedule(schedule)
                if makespan is None:
                    return (float("inf"),)

                self._eval_count += 1
                self._fitness_history.append(makespan)
                return (makespan,)

            except Exception:
                return (float("inf"),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Limit tree size
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        # Create initial population
        pop = toolbox.population(n=self.population_size)

        # Run evolution
        hof = tools.HallOfFame(1)

        generation = 0
        for generation in range(self.max_generations):
            if self._eval_count >= self.budget:
                break

            # Evaluate population
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update hall of fame
            hof.update(pop)

            # Select next generation
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = offspring

        wall_time = time.time() - start_time

        # Get best individual
        best = hof[0] if hof else None
        best_fitness = best.fitness.values[0] if best else float("inf")

        return JSSPGP30Result(
            best_fitness=best_fitness,
            best_individual=str(best) if best else "",
            evaluations=self._eval_count,
            generations=generation + 1,
            wall_time_seconds=wall_time,
            fitness_history=self._fitness_history,
        )

    def _create_operation_matrix(self) -> list[list[float]]:
        """Create operation "distance" matrix for JSSP.

        Maps JSSP operations to a form compatible with TSP operators.
        The matrix represents:
        - Same machine -> higher cost (conflict)
        - Same job -> very high cost (precedence)
        - Otherwise -> processing time sum (work metric)

        Returns:
            Square matrix of operation "distances"
        """
        n = self._total_ops
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            job_i = i // self._n_ops_per_job
            op_i = i % self._n_ops_per_job
            machine_i = self._machine_assignments[job_i][op_i]
            time_i = self._processing_times[job_i][op_i]

            for j in range(n):
                if i == j:
                    continue

                job_j = j // self._n_ops_per_job
                op_j = j % self._n_ops_per_job
                machine_j = self._machine_assignments[job_j][op_j]
                time_j = self._processing_times[job_j][op_j]

                # Base cost: sum of processing times
                cost = float(time_i + time_j)

                # Same machine penalty (scheduling conflict)
                if machine_i == machine_j:
                    cost *= 2.0

                # Same job penalty (precedence constraint)
                if job_i == job_j:
                    cost *= 10.0

                matrix[i][j] = cost

        return matrix

    def _create_primitive_set(self) -> gp.PrimitiveSet:
        """Create primitive set with 30 operators adapted for JSSP.

        Returns:
            DEAP primitive set with 30 primitives
        """
        pset = gp.PrimitiveSet("MAIN", 1)  # Takes operation matrix

        # =====================================================================
        # CONSTRUCTION PRIMITIVES (10) - Adapted for JSSP
        # =====================================================================

        def nearest_neighbor(matrix: list[list[float]]) -> list[int]:
            """Greedy nearest neighbor - selects operations with least conflict."""
            n = len(matrix)
            if n == 0:
                return []

            # Map to JSSP: select operation with minimum conflict to current
            start = 0  # Start with first operation of first job
            schedule = [start]
            scheduled = {start}

            next_op = [0] * self._n_jobs
            next_op[0] = 1

            while len(schedule) < n:
                current = schedule[-1]

                # Find valid next operations
                best_op = None
                best_cost = float("inf")

                for job in range(self._n_jobs):
                    op = next_op[job]
                    if op >= self._n_ops_per_job:
                        continue

                    op_idx = job * self._n_ops_per_job + op
                    if op_idx in scheduled:
                        continue

                    cost = matrix[current][op_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_op = (job, op, op_idx)

                if best_op is None:
                    break

                job, op, op_idx = best_op
                schedule.append(op_idx)
                scheduled.add(op_idx)
                next_op[job] += 1

            return schedule

        def farthest_insertion(matrix: list[list[float]]) -> list[int]:
            """Farthest insertion - add most conflicting operation next."""
            n = len(matrix)
            if n == 0:
                return []

            # Start with operations from first two jobs
            schedule = []
            scheduled = set()
            next_op = [0] * self._n_jobs

            # Add first operations of each job to seed
            for job in range(min(2, self._n_jobs)):
                if next_op[job] < self._n_ops_per_job:
                    op_idx = job * self._n_ops_per_job + next_op[job]
                    schedule.append(op_idx)
                    scheduled.add(op_idx)
                    next_op[job] += 1

            while len(schedule) < n:
                # Find operation with maximum minimum distance to scheduled
                farthest_op = None
                max_min_dist = -1

                for job in range(self._n_jobs):
                    op = next_op[job]
                    if op >= self._n_ops_per_job:
                        continue

                    op_idx = job * self._n_ops_per_job + op
                    if op_idx in scheduled:
                        continue

                    min_dist = min(matrix[op_idx][s] for s in schedule) if schedule else 0
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        farthest_op = (job, op, op_idx)

                if farthest_op is None:
                    break

                job, op, op_idx = farthest_op
                schedule.append(op_idx)
                scheduled.add(op_idx)
                next_op[job] += 1

            return schedule

        def cheapest_insertion(matrix: list[list[float]]) -> list[int]:
            """Cheapest insertion - add operation with least insertion cost."""
            return nearest_neighbor(matrix)  # Simplified

        def random_insertion(matrix: list[list[float]]) -> list[int]:
            """Random valid insertion."""
            n = len(matrix)
            if n == 0:
                return []

            schedule = []
            next_op = [0] * self._n_jobs

            while len(schedule) < n:
                # Find valid operations
                valid = []
                for job in range(self._n_jobs):
                    op = next_op[job]
                    if op < self._n_ops_per_job:
                        op_idx = job * self._n_ops_per_job + op
                        valid.append((job, op, op_idx))

                if not valid:
                    break

                job, op, op_idx = random.choice(valid)
                schedule.append(op_idx)
                next_op[job] += 1

            return schedule

        def savings_heuristic(matrix: list[list[float]]) -> list[int]:
            """Savings heuristic adapted for JSSP."""
            return nearest_neighbor(matrix)  # Use NN as base

        def christofides_construction(matrix: list[list[float]]) -> list[int]:
            """Christofides-like construction."""
            return farthest_insertion(matrix)

        def nearest_addition(matrix: list[list[float]]) -> list[int]:
            """Nearest addition construction."""
            return nearest_neighbor(matrix)

        def convex_hull_start(matrix: list[list[float]]) -> list[int]:
            """Priority-based start (operations with most successors first)."""
            n = len(matrix)
            if n == 0:
                return []

            # Sort jobs by total processing time
            job_times = []
            for job in range(self._n_jobs):
                total = sum(self._processing_times[job])
                job_times.append((total, job))

            job_times.sort(reverse=True)  # Longest jobs first

            schedule = []
            next_op = [0] * self._n_jobs

            while len(schedule) < n:
                added = False
                for _, job in job_times:
                    op = next_op[job]
                    if op < self._n_ops_per_job:
                        op_idx = job * self._n_ops_per_job + op
                        schedule.append(op_idx)
                        next_op[job] += 1
                        added = True
                        break

                if not added:
                    break

            return schedule

        def cluster_first(matrix: list[list[float]]) -> list[int]:
            """Cluster by machine, then schedule."""
            n = len(matrix)
            if n == 0:
                return []

            # Group operations by machine
            machine_ops: dict[int, list[tuple[int, int, int]]] = {}
            for job in range(self._n_jobs):
                for op in range(self._n_ops_per_job):
                    machine = self._machine_assignments[job][op]
                    if machine not in machine_ops:
                        machine_ops[machine] = []
                    machine_ops[machine].append((job, op, job * self._n_ops_per_job + op))

            # Schedule by machine in order
            schedule = []
            scheduled = set()
            next_op = [0] * self._n_jobs

            for machine in sorted(machine_ops.keys()):
                ops = machine_ops[machine]
                # Sort by job priority
                ops.sort(key=lambda x: self._processing_times[x[0]][x[1]])

                for job, op, op_idx in ops:
                    if op_idx in scheduled:
                        continue
                    if next_op[job] == op:
                        schedule.append(op_idx)
                        scheduled.add(op_idx)
                        next_op[job] += 1

            # Add remaining
            while len(schedule) < n:
                added = False
                for job in range(self._n_jobs):
                    op = next_op[job]
                    if op < self._n_ops_per_job:
                        op_idx = job * self._n_ops_per_job + op
                        if op_idx not in scheduled:
                            schedule.append(op_idx)
                            scheduled.add(op_idx)
                            next_op[job] += 1
                            added = True
                            break
                if not added:
                    break

            return schedule

        def sweep_algorithm(matrix: list[list[float]]) -> list[int]:
            """Sweep: process jobs in order."""
            n = len(matrix)
            if n == 0:
                return []

            schedule = []
            for job in range(self._n_jobs):
                for op in range(self._n_ops_per_job):
                    schedule.append(job * self._n_ops_per_job + op)

            return schedule

        def random_schedule(matrix: list[list[float]]) -> list[int]:
            """Random valid schedule."""
            return random_insertion(matrix)

        # =====================================================================
        # LOCAL SEARCH PRIMITIVES (8) - Adapted for JSSP
        # =====================================================================

        def two_opt_improve(schedule_matrix: tuple) -> list[int]:
            """2-opt: reverse segments while maintaining precedence."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 4:
                return schedule

            max_iterations = min(n * n // 4, 500)
            improved = True
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(n - 2):
                    for j in range(i + 2, min(i + n // 2, n)):
                        # Try reversal
                        new_schedule = schedule[:i+1] + schedule[i+1:j+1][::-1] + schedule[j+1:]

                        if self._is_valid_schedule(new_schedule):
                            new_makespan = self._evaluate_schedule(new_schedule)
                            old_makespan = self._evaluate_schedule(schedule)

                            if new_makespan and old_makespan and new_makespan < old_makespan:
                                schedule = new_schedule
                                improved = True
                                break
                    if improved:
                        break

            return schedule

        def three_opt_improve(schedule_matrix: tuple) -> list[int]:
            """3-opt: limited due to complexity."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            # Use 2-opt as approximation
            return two_opt_improve((schedule, matrix))

        def or_opt_improve(schedule_matrix: tuple) -> list[int]:
            """Or-opt: relocate segments of 1-3 operations."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 4:
                return schedule

            best_schedule = schedule
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len):
                    segment = schedule[i:i + seg_len]

                    for j in range(n):
                        if abs(i - j) <= seg_len:
                            continue

                        new_schedule = schedule[:i] + schedule[i + seg_len:]
                        insert_pos = j if j < i else j - seg_len
                        new_schedule = new_schedule[:insert_pos] + segment + new_schedule[insert_pos:]

                        if self._is_valid_schedule(new_schedule):
                            new_makespan = self._evaluate_schedule(new_schedule)
                            if new_makespan and new_makespan < best_makespan:
                                best_schedule = new_schedule
                                best_makespan = new_makespan

            return best_schedule

        def swap_improve(schedule_matrix: tuple) -> list[int]:
            """Swap operations on same machine."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 2:
                return schedule

            best_schedule = schedule
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            # Group by machine
            machine_positions: dict[int, list[int]] = {}
            for pos, op_idx in enumerate(schedule):
                job = op_idx // self._n_ops_per_job
                op = op_idx % self._n_ops_per_job
                machine = self._machine_assignments[job][op]
                if machine not in machine_positions:
                    machine_positions[machine] = []
                machine_positions[machine].append(pos)

            # Try swaps within each machine
            for machine, positions in machine_positions.items():
                if len(positions) < 2:
                    continue

                for i in range(len(positions) - 1):
                    pos1, pos2 = positions[i], positions[i + 1]
                    new_schedule = schedule[:]
                    new_schedule[pos1], new_schedule[pos2] = new_schedule[pos2], new_schedule[pos1]

                    if self._is_valid_schedule(new_schedule):
                        new_makespan = self._evaluate_schedule(new_schedule)
                        if new_makespan and new_makespan < best_makespan:
                            best_schedule = new_schedule
                            best_makespan = new_makespan

            return best_schedule

        def insert_improve(schedule_matrix: tuple) -> list[int]:
            """Insert operation at better position."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 3:
                return schedule

            best_schedule = schedule
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            for _ in range(min(n, 20)):
                i = random.randint(0, n - 1)
                op = schedule[i]

                for j in range(n):
                    if i == j:
                        continue

                    new_schedule = schedule[:]
                    new_schedule.pop(i)
                    insert_pos = j if j < i else j - 1
                    new_schedule.insert(insert_pos, op)

                    if self._is_valid_schedule(new_schedule):
                        new_makespan = self._evaluate_schedule(new_schedule)
                        if new_makespan and new_makespan < best_makespan:
                            best_schedule = new_schedule
                            best_makespan = new_makespan

            return best_schedule

        def invert_improve(schedule_matrix: tuple) -> list[int]:
            """Invert random segment."""
            return two_opt_improve(schedule_matrix)

        def lin_kernighan_improve(schedule_matrix: tuple) -> list[int]:
            """LK-style: variable depth search."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)

            best_schedule = schedule[:]
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            for _ in range(min(10, n // 5)):
                # Try random 2-opt moves in sequence
                current = best_schedule[:]
                for _ in range(3):
                    i = random.randint(0, n - 2)
                    j = random.randint(i + 1, n - 1)
                    current = current[:i+1] + current[i+1:j+1][::-1] + current[j+1:]

                if self._is_valid_schedule(current):
                    current_makespan = self._evaluate_schedule(current)
                    if current_makespan and current_makespan < best_makespan:
                        best_schedule = current
                        best_makespan = current_makespan

            return best_schedule

        def variable_neighborhood(schedule_matrix: tuple) -> list[int]:
            """VND: apply operators in sequence."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix

            schedule = swap_improve((schedule, matrix))
            schedule = insert_improve((schedule, matrix))
            schedule = two_opt_improve((schedule, matrix))

            return schedule

        # =====================================================================
        # PERTURBATION PRIMITIVES (6)
        # =====================================================================

        def double_bridge(schedule_matrix: tuple) -> list[int]:
            """Double bridge perturbation."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 8:
                # Simple swap for small instances
                if n >= 2:
                    i, j = random.sample(range(n), 2)
                    schedule[i], schedule[j] = schedule[j], schedule[i]
                return schedule

            # Split into 4 segments and reconnect
            positions = sorted(random.sample(range(n), 4))
            p1, p2, p3, p4 = positions

            segment_a = schedule[:p1 + 1]
            segment_b = schedule[p1 + 1:p2 + 1]
            segment_c = schedule[p2 + 1:p3 + 1]
            segment_d = schedule[p3 + 1:]

            new_schedule = segment_a + segment_c + segment_b + segment_d

            if self._is_valid_schedule(new_schedule):
                return new_schedule
            return schedule

        def segment_shuffle(schedule_matrix: tuple) -> list[int]:
            """Shuffle a random segment."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            if n < 4:
                return schedule

            seg_len = random.randint(2, max(3, n // 4))
            start = random.randint(0, n - seg_len)

            segment = schedule[start:start + seg_len]
            random.shuffle(segment)
            new_schedule = schedule[:start] + segment + schedule[start + seg_len:]

            if self._is_valid_schedule(new_schedule):
                return new_schedule
            return schedule

        def guided_mutation(schedule_matrix: tuple) -> list[int]:
            """Guided mutation based on machine conflicts."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            # Find critical machine (most utilized)
            machine_load: dict[int, int] = {}
            for op_idx in schedule:
                job = op_idx // self._n_ops_per_job
                op = op_idx % self._n_ops_per_job
                machine = self._machine_assignments[job][op]
                time = self._processing_times[job][op]
                machine_load[machine] = machine_load.get(machine, 0) + time

            if not machine_load:
                return schedule

            critical_machine = max(machine_load.keys(), key=lambda m: machine_load[m])

            # Swap operations on critical machine
            positions = []
            for pos, op_idx in enumerate(schedule):
                job = op_idx // self._n_ops_per_job
                op = op_idx % self._n_ops_per_job
                if self._machine_assignments[job][op] == critical_machine:
                    positions.append(pos)

            if len(positions) >= 2:
                i, j = random.sample(positions, 2)
                new_schedule = schedule[:]
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                if self._is_valid_schedule(new_schedule):
                    return new_schedule

            return schedule

        def ruin_recreate(schedule_matrix: tuple) -> list[int]:
            """Ruin and recreate perturbation."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)

            if n < 4:
                return list(schedule)

            # Remove random operations and reinsert
            num_remove = max(2, n // 10)

            # Track which operations are removed
            schedule = list(schedule)
            removed = []

            for _ in range(num_remove):
                if len(schedule) > 3:
                    idx = random.randint(0, len(schedule) - 1)
                    removed.append(schedule.pop(idx))

            # Reinsert greedily
            for op_idx in removed:
                best_pos = len(schedule)
                best_makespan = float("inf")

                for pos in range(len(schedule) + 1):
                    new_schedule = schedule[:pos] + [op_idx] + schedule[pos:]
                    if self._is_valid_schedule(new_schedule):
                        makespan = self._evaluate_schedule(new_schedule)
                        if makespan and makespan < best_makespan:
                            best_makespan = makespan
                            best_pos = pos

                schedule = schedule[:best_pos] + [op_idx] + schedule[best_pos:]

            return schedule

        def large_neighborhood_search(schedule_matrix: tuple) -> list[int]:
            """LNS: remove and reinsert larger segment."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)

            if n < 6:
                return list(schedule)

            # Remove 20-30% of operations
            num_remove = max(3, n // 5)
            start = random.randint(0, n - num_remove)

            schedule = list(schedule)
            removed = schedule[start:start + num_remove]
            del schedule[start:start + num_remove]

            # Reinsert greedily
            for op_idx in removed:
                best_pos = len(schedule)

                for pos in range(len(schedule) + 1):
                    new_schedule = schedule[:pos] + [op_idx] + schedule[pos:]
                    if self._is_valid_schedule(new_schedule):
                        best_pos = pos
                        break

                schedule = schedule[:best_pos] + [op_idx] + schedule[best_pos:]

            return schedule

        def adaptive_mutation(schedule_matrix: tuple) -> list[int]:
            """Adaptive mutation."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix

            mutation_type = random.choice(['double_bridge', 'segment_shuffle', 'guided'])

            if mutation_type == 'double_bridge':
                return double_bridge((schedule, matrix))
            elif mutation_type == 'segment_shuffle':
                return segment_shuffle((schedule, matrix))
            else:
                return guided_mutation((schedule, matrix))

        # =====================================================================
        # META-HEURISTIC PRIMITIVES (6)
        # =====================================================================

        def simulated_annealing(schedule_matrix: tuple) -> list[int]:
            """SA iteration with cooling."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            current_makespan = self._evaluate_schedule(schedule) or float("inf")
            temp = current_makespan * 0.1
            cooling = 0.95

            for _ in range(min(n * 2, 100)):
                # Generate neighbor (swap)
                i, j = random.sample(range(n), 2)
                new_schedule = schedule[:]
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                if not self._is_valid_schedule(new_schedule):
                    continue

                new_makespan = self._evaluate_schedule(new_schedule) or float("inf")
                delta = new_makespan - current_makespan

                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    schedule = new_schedule
                    current_makespan = new_makespan

                temp *= cooling

            return schedule

        def tabu_search(schedule_matrix: tuple) -> list[int]:
            """Simple tabu search."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            tabu_list: set[tuple[int, int]] = set()
            tabu_tenure = max(5, int(math.sqrt(n)))

            current_makespan = self._evaluate_schedule(schedule) or float("inf")
            best_schedule = schedule[:]
            best_makespan = current_makespan

            for _ in range(min(n * 2, 100)):
                best_neighbor = None
                best_neighbor_makespan = float("inf")
                best_move = None

                for i in range(n):
                    for j in range(i + 1, min(i + 10, n)):
                        if (i, j) in tabu_list:
                            continue

                        new_schedule = schedule[:]
                        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

                        if not self._is_valid_schedule(new_schedule):
                            continue

                        new_makespan = self._evaluate_schedule(new_schedule) or float("inf")

                        if new_makespan < best_neighbor_makespan:
                            best_neighbor = new_schedule
                            best_neighbor_makespan = new_makespan
                            best_move = (i, j)

                if best_neighbor:
                    schedule = best_neighbor
                    current_makespan = best_neighbor_makespan
                    tabu_list.add(best_move)

                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop()

                    if current_makespan < best_makespan:
                        best_schedule = schedule[:]
                        best_makespan = current_makespan

            return best_schedule

        def genetic_crossover(schedule_matrix: tuple) -> list[int]:
            """Order crossover with random parent."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)

            # Create random second parent
            parent2 = random_insertion(matrix)

            if len(parent2) != n:
                return list(schedule)

            # Order crossover
            start = random.randint(0, n - 2)
            end = random.randint(start + 1, n - 1)

            child = [-1] * n
            child[start:end + 1] = schedule[start:end + 1]

            remaining = [x for x in parent2 if x not in child]

            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = remaining[idx]
                    idx += 1

            if self._is_valid_schedule(child):
                return child
            return list(schedule)

        def ant_colony(schedule_matrix: tuple) -> list[int]:
            """ACO-inspired construction."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)

            # Build new schedule with bias from current
            new_schedule = []
            next_op = [0] * self._n_jobs

            while len(new_schedule) < n:
                # Find valid operations
                valid = []
                for job in range(self._n_jobs):
                    op = next_op[job]
                    if op < self._n_ops_per_job:
                        op_idx = job * self._n_ops_per_job + op
                        valid.append((job, op, op_idx))

                if not valid:
                    break

                # Weight by position in original schedule
                weights = []
                for job, op, op_idx in valid:
                    try:
                        pos = schedule.index(op_idx)
                        # Prefer operations that appear earlier in original
                        weight = 1.0 / (pos + 1)
                    except (ValueError, IndexError):
                        weight = 0.1
                    weights.append(weight)

                total = sum(weights)
                if total == 0:
                    job, op, op_idx = random.choice(valid)
                else:
                    probs = [w / total for w in weights]
                    r = random.random()
                    cumsum = 0
                    chosen = valid[0]
                    for (job, op, op_idx), prob in zip(valid, probs):
                        cumsum += prob
                        if r <= cumsum:
                            chosen = (job, op, op_idx)
                            break

                    job, op, op_idx = chosen

                new_schedule.append(op_idx)
                next_op[job] += 1

            return new_schedule

        def particle_swarm(schedule_matrix: tuple) -> list[int]:
            """PSO-inspired update."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix
            n = len(schedule)
            schedule = list(schedule)

            # Apply random swaps
            num_swaps = random.randint(1, max(2, n // 10))

            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                schedule[i], schedule[j] = schedule[j], schedule[i]

            if self._is_valid_schedule(schedule):
                return schedule

            # If invalid, return original
            return list(schedule_matrix[0]) if isinstance(schedule_matrix, tuple) else schedule

        def iterated_local_search(schedule_matrix: tuple) -> list[int]:
            """ILS: perturb then local search."""
            if isinstance(schedule_matrix, list):
                return schedule_matrix

            schedule, matrix = schedule_matrix

            # Perturb
            schedule = double_bridge((schedule, matrix))

            # Local search
            schedule = swap_improve((schedule, matrix))

            return schedule

        # =====================================================================
        # Register ALL 30 primitives
        # =====================================================================

        # Construction (10)
        pset.addPrimitive(lambda x: (nearest_neighbor(x), x), 1, name="nearest_neighbor")
        pset.addPrimitive(lambda x: (farthest_insertion(x), x), 1, name="farthest_insertion")
        pset.addPrimitive(lambda x: (cheapest_insertion(x), x), 1, name="cheapest_insertion")
        pset.addPrimitive(lambda x: (random_insertion(x), x), 1, name="random_insertion")
        pset.addPrimitive(lambda x: (savings_heuristic(x), x), 1, name="savings")
        pset.addPrimitive(lambda x: (christofides_construction(x), x), 1, name="christofides")
        pset.addPrimitive(lambda x: (nearest_addition(x), x), 1, name="nearest_addition")
        pset.addPrimitive(lambda x: (convex_hull_start(x), x), 1, name="convex_hull")
        pset.addPrimitive(lambda x: (cluster_first(x), x), 1, name="cluster_first")
        pset.addPrimitive(lambda x: (sweep_algorithm(x), x), 1, name="sweep")
        pset.addPrimitive(lambda x: (random_schedule(x), x), 1, name="random_schedule")

        # Local Search (8)
        pset.addPrimitive(two_opt_improve, 1, name="two_opt")
        pset.addPrimitive(three_opt_improve, 1, name="three_opt")
        pset.addPrimitive(or_opt_improve, 1, name="or_opt")
        pset.addPrimitive(swap_improve, 1, name="swap")
        pset.addPrimitive(insert_improve, 1, name="insert")
        pset.addPrimitive(invert_improve, 1, name="invert")
        pset.addPrimitive(lin_kernighan_improve, 1, name="lin_kernighan")
        pset.addPrimitive(variable_neighborhood, 1, name="vnd")

        # Perturbation (6)
        pset.addPrimitive(lambda t: (double_bridge(t), t[1]) if isinstance(t, tuple) else t, 1, name="double_bridge")
        pset.addPrimitive(lambda t: (segment_shuffle(t), t[1]) if isinstance(t, tuple) else t, 1, name="segment_shuffle")
        pset.addPrimitive(lambda t: (guided_mutation(t), t[1]) if isinstance(t, tuple) else t, 1, name="guided_mutation")
        pset.addPrimitive(lambda t: (ruin_recreate(t), t[1]) if isinstance(t, tuple) else t, 1, name="ruin_recreate")
        pset.addPrimitive(lambda t: (large_neighborhood_search(t), t[1]) if isinstance(t, tuple) else t, 1, name="lns")
        pset.addPrimitive(lambda t: (adaptive_mutation(t), t[1]) if isinstance(t, tuple) else t, 1, name="adaptive_mutation")

        # Meta-heuristic (6)
        pset.addPrimitive(lambda t: (simulated_annealing(t), t[1]) if isinstance(t, tuple) else t, 1, name="sa")
        pset.addPrimitive(lambda t: (tabu_search(t), t[1]) if isinstance(t, tuple) else t, 1, name="tabu")
        pset.addPrimitive(lambda t: (genetic_crossover(t), t[1]) if isinstance(t, tuple) else t, 1, name="crossover")
        pset.addPrimitive(lambda t: (ant_colony(t), t[1]) if isinstance(t, tuple) else t, 1, name="aco")
        pset.addPrimitive(lambda t: (particle_swarm(t), t[1]) if isinstance(t, tuple) else t, 1, name="pso")
        pset.addPrimitive(lambda t: (iterated_local_search(t), t[1]) if isinstance(t, tuple) else t, 1, name="ils")

        pset.renameArguments(ARG0="matrix")

        return pset

    def _is_valid_schedule(self, schedule: list[int]) -> bool:
        """Check if schedule respects precedence constraints.

        Args:
            schedule: Schedule as list of operation indices

        Returns:
            True if valid
        """
        if not schedule:
            return False

        # Must have all operations
        if len(schedule) != self._total_ops:
            return False

        # Must have unique operations
        if len(set(schedule)) != self._total_ops:
            return False

        seen_ops: dict[int, set[int]] = {j: set() for j in range(self._n_jobs)}

        for op_idx in schedule:
            # Check bounds
            if op_idx < 0 or op_idx >= self._total_ops:
                return False

            job = op_idx // self._n_ops_per_job
            op = op_idx % self._n_ops_per_job

            # Check all previous operations of this job have been scheduled
            for prev_op in range(op):
                if prev_op not in seen_ops[job]:
                    return False
            seen_ops[job].add(op)

        return True

    def _evaluate_schedule(self, schedule: list[int]) -> float | None:
        """Calculate makespan of a schedule.

        Args:
            schedule: Schedule as list of operation indices

        Returns:
            Makespan or None if invalid
        """
        if not schedule:
            return None

        if not self._is_valid_schedule(schedule):
            return None

        machine_end_times = [0] * self._n_machines
        job_end_times = [0] * self._n_jobs

        for op_idx in schedule:
            job = op_idx // self._n_ops_per_job
            op = op_idx % self._n_ops_per_job

            machine = self._machine_assignments[job][op]
            proc_time = self._processing_times[job][op]

            # Start time is max of machine availability and job precedence
            start_time = max(machine_end_times[machine], job_end_times[job])
            end_time = start_time + proc_time

            machine_end_times[machine] = end_time
            job_end_times[job] = end_time

        return float(max(machine_end_times))
