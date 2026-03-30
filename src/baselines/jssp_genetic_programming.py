"""Genetic Programming baseline for JSSP using DEAP.

Standard GP baseline for comparison with NS-SE on Job Shop Scheduling.
Uses same budget for fair comparison.
"""

import operator
import random
import time
from typing import Any

from deap import algorithms, base, creator, gp, tools
from pydantic import BaseModel, Field


class JSSPGPResult(BaseModel):
    """Result from GP run on JSSP."""

    best_fitness: float
    best_individual: str = ""
    evaluations: int = 0
    generations: int = 0
    wall_time_seconds: float = 0.0
    fitness_history: list[float] = Field(default_factory=list)


class JSSPGeneticProgramming:
    """Genetic Programming for JSSP using DEAP.

    Evolves programs that construct JSSP schedules by combining
    priority rules and local search operators.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        budget: int = 1000,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        max_tree_depth: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize GP baseline for JSSP.

        Args:
            population_size: Size of population
            max_generations: Maximum generations
            budget: Maximum fitness evaluations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            tournament_size: Tournament selection size
            max_tree_depth: Maximum tree depth
            seed: Random seed
        """
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
    ) -> JSSPGPResult:
        """Run GP optimization for JSSP.

        Args:
            processing_times: Processing time for each operation [job][op]
            machine_assignments: Machine for each operation [job][op]

        Returns:
            JSSPGPResult with best solution and statistics
        """
        start_time = time.time()
        n_jobs = len(processing_times)
        n_machines = len(set(m for job in machine_assignments for m in job))
        n_ops_per_job = len(processing_times[0]) if processing_times else 0

        self._eval_count = 0
        self._fitness_history = []

        # Store problem data for evaluation
        self._processing_times = processing_times
        self._machine_assignments = machine_assignments
        self._n_jobs = n_jobs
        self._n_machines = n_machines
        self._n_ops_per_job = n_ops_per_job

        # Define primitive set for JSSP scheduling
        pset = self._create_primitive_set()

        # Create fitness and individual types
        if not hasattr(creator, "JSSPFitnessMin"):
            creator.create("JSSPFitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "JSSPIndividual"):
            creator.create("JSSPIndividual", gp.PrimitiveTree, fitness=creator.JSSPFitnessMin)

        # Create toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=self.max_tree_depth)
        toolbox.register("individual", tools.initIterate, creator.JSSPIndividual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Evaluation function
        def evaluate(individual: Any) -> tuple[float]:
            if self._eval_count >= self.budget:
                return (float("inf"),)

            try:
                func = toolbox.compile(expr=individual)
                schedule = func()

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

        return JSSPGPResult(
            best_fitness=best_fitness,
            best_individual=str(best) if best else "",
            evaluations=self._eval_count,
            generations=generation + 1,
            wall_time_seconds=wall_time,
            fitness_history=self._fitness_history,
        )

    def _create_primitive_set(self) -> gp.PrimitiveSet:
        """Create primitive set for JSSP.

        Returns:
            DEAP primitive set
        """
        pset = gp.PrimitiveSet("MAIN", 0)

        # Priority rule terminals - return schedules
        def spt_schedule() -> list[tuple[int, int]]:
            """Shortest Processing Time first."""
            return self._build_schedule_with_rule("spt")

        def lpt_schedule() -> list[tuple[int, int]]:
            """Longest Processing Time first."""
            return self._build_schedule_with_rule("lpt")

        def fifo_schedule() -> list[tuple[int, int]]:
            """First In First Out (by job index)."""
            return self._build_schedule_with_rule("fifo")

        def random_schedule() -> list[tuple[int, int]]:
            """Random priority."""
            return self._build_schedule_with_rule("random")

        def mwkr_schedule() -> list[tuple[int, int]]:
            """Most Work Remaining."""
            return self._build_schedule_with_rule("mwkr")

        # Local search operators
        def swap_improve(schedule: list[tuple[int, int]]) -> list[tuple[int, int]]:
            """Try swapping adjacent operations on same machine."""
            if not schedule or not isinstance(schedule, list):
                return schedule

            schedule = list(schedule)
            n = len(schedule)
            if n < 2:
                return schedule

            # Group by machine
            machine_ops: dict[int, list[int]] = {}
            for idx, (job, op) in enumerate(schedule):
                machine = self._machine_assignments[job][op]
                if machine not in machine_ops:
                    machine_ops[machine] = []
                machine_ops[machine].append(idx)

            # Try swaps on each machine (limited iterations)
            best_schedule = schedule
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            for machine, ops in machine_ops.items():
                if len(ops) < 2:
                    continue
                for i in range(min(len(ops) - 1, 5)):  # Limit swaps
                    new_schedule = best_schedule.copy()
                    idx1, idx2 = ops[i], ops[i + 1]
                    new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]

                    # Check if valid (precedence constraints)
                    if self._is_valid_schedule(new_schedule):
                        new_makespan = self._evaluate_schedule(new_schedule)
                        if new_makespan and new_makespan < best_makespan:
                            best_schedule = new_schedule
                            best_makespan = new_makespan

            return best_schedule

        def insert_improve(schedule: list[tuple[int, int]]) -> list[tuple[int, int]]:
            """Try inserting operations at different positions."""
            if not schedule or not isinstance(schedule, list):
                return schedule

            schedule = list(schedule)
            n = len(schedule)
            if n < 3:
                return schedule

            best_schedule = schedule
            best_makespan = self._evaluate_schedule(schedule) or float("inf")

            # Try a few random insertions
            for _ in range(min(10, n)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue

                new_schedule = schedule.copy()
                op = new_schedule.pop(i)
                new_schedule.insert(j, op)

                if self._is_valid_schedule(new_schedule):
                    new_makespan = self._evaluate_schedule(new_schedule)
                    if new_makespan and new_makespan < best_makespan:
                        best_schedule = new_schedule
                        best_makespan = new_makespan

            return best_schedule

        # Add terminals (scheduling rules)
        pset.addTerminal(spt_schedule, name="spt")
        pset.addTerminal(lpt_schedule, name="lpt")
        pset.addTerminal(fifo_schedule, name="fifo")
        pset.addTerminal(random_schedule, name="random")
        pset.addTerminal(mwkr_schedule, name="mwkr")

        # Add primitives (improvement operators)
        pset.addPrimitive(swap_improve, 1, name="swap")
        pset.addPrimitive(insert_improve, 1, name="insert")

        return pset

    def _build_schedule_with_rule(self, rule: str) -> list[tuple[int, int]]:
        """Build a schedule using a priority rule.

        Args:
            rule: Priority rule name (spt, lpt, fifo, random, mwkr)

        Returns:
            Schedule as list of (job, operation) tuples
        """
        schedule: list[tuple[int, int]] = []
        next_op = [0] * self._n_jobs  # Next operation index for each job
        machine_available = [0] * self._n_machines
        job_available = [0] * self._n_jobs

        total_ops = self._n_jobs * self._n_ops_per_job

        while len(schedule) < total_ops:
            # Find ready operations
            ready = []
            for job in range(self._n_jobs):
                op = next_op[job]
                if op < self._n_ops_per_job:
                    ready.append((job, op))

            if not ready:
                break

            # Select based on rule
            if rule == "spt":
                ready.sort(key=lambda x: self._processing_times[x[0]][x[1]])
                selected = ready[0]
            elif rule == "lpt":
                ready.sort(key=lambda x: -self._processing_times[x[0]][x[1]])
                selected = ready[0]
            elif rule == "fifo":
                selected = ready[0]  # First job first
            elif rule == "mwkr":
                # Most work remaining
                def work_remaining(x):
                    job, op = x
                    return sum(self._processing_times[job][op:])
                ready.sort(key=work_remaining, reverse=True)
                selected = ready[0]
            else:  # random
                selected = random.choice(ready)

            job, op = selected
            schedule.append((job, op))
            next_op[job] += 1

        return schedule

    def _is_valid_schedule(self, schedule: list[tuple[int, int]]) -> bool:
        """Check if schedule respects precedence constraints.

        Args:
            schedule: Schedule as list of (job, operation) tuples

        Returns:
            True if valid
        """
        seen_ops: dict[int, set[int]] = {j: set() for j in range(self._n_jobs)}

        for job, op in schedule:
            # Check all previous operations of this job have been scheduled
            for prev_op in range(op):
                if prev_op not in seen_ops[job]:
                    return False
            seen_ops[job].add(op)

        return True

    def _evaluate_schedule(self, schedule: list[tuple[int, int]]) -> float | None:
        """Calculate makespan of a schedule.

        Args:
            schedule: Schedule as list of (job, operation) tuples

        Returns:
            Makespan or None if invalid
        """
        if not schedule:
            return None

        if not self._is_valid_schedule(schedule):
            return None

        machine_end_times = [0] * self._n_machines
        job_end_times = [0] * self._n_jobs

        for job, op in schedule:
            machine = self._machine_assignments[job][op]
            proc_time = self._processing_times[job][op]

            # Start time is max of machine availability and job precedence
            start_time = max(machine_end_times[machine], job_end_times[job])
            end_time = start_time + proc_time

            machine_end_times[machine] = end_time
            job_end_times[job] = end_time

        return max(machine_end_times)
