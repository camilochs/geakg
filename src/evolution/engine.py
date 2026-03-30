"""NS-SE Evolution Engine.

Main orchestrator that combines all components:
- AKG for structured algorithm representation
- ACO/MMAS for operator selection (no LLM at runtime)
- Population management for evolution

The LLM is only used OFFLINE to construct the AKG topology.
At runtime, MMAS traverses the AKG without any LLM calls.
"""

import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.geakg.aco import ACOSelector, ACOConfig
from src.geakg.conditions import ExecutionContext
from src.geakg.graph import AlgorithmicKnowledgeGraph, Trajectory
from src.geakg.nodes import OperatorCategory
from src.evolution.population import Algorithm, Population, PopulationManager
from src.retrieval.lookup import TrajectoryLookup


class EngineConfig(BaseModel):
    """Configuration for NS-SE engine."""

    # Population settings
    population_size: int = Field(default=20, ge=5)
    elite_count: int = Field(default=2, ge=1)

    # Algorithm settings
    min_operators: int = Field(default=2, ge=1)
    max_operators: int = Field(default=6, ge=2)

    # Evolution settings
    max_generations: int = Field(default=50, ge=1)
    max_evaluations: int = Field(default=1000, ge=10)
    convergence_generations: int = Field(default=20, ge=5)
    convergence_threshold: float = Field(default=0.01, ge=0.0)

    # ACO/MMAS settings
    aco_config: ACOConfig | None = Field(default=None)

    # Logging
    log_all_trajectories: bool = Field(default=True)
    verbose: bool = Field(default=True)


class EvolutionStats(BaseModel):
    """Statistics from evolution run."""

    generations: int = 0
    evaluations: int = 0
    aco_solutions_constructed: int = 0
    pheromone_updates: int = 0
    best_fitness: float = float("inf")
    best_fitness_history: list[float] = Field(default_factory=list)
    avg_fitness_history: list[float] = Field(default_factory=list)
    wall_time_seconds: float = 0.0


class NSGGEEngine:
    """Neuro-Symbolic Graph-based Genetic Evolution Engine.

    Orchestrates the complete NS-SE pipeline with MMAS-only operator selection:
    1. Initialize population with random algorithms
    2. Evaluate fitness on problem instances
    3. Select parents
    4. Generate offspring using ACO/MMAS traversal of AKG
    5. Replace population with elitist strategy
    6. Update pheromones based on best solutions
    7. Record successful trajectories

    NOTE: The LLM is only used OFFLINE to construct the AKG topology.
    At runtime, MMAS traverses the AKG without any LLM calls.
    """

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        fitness_function: Callable[[Algorithm, Any], float],
        problem_instance: Any,
        problem_type: str = "tsp",
        config: EngineConfig | None = None,
        trajectory_db_path: Path | str | None = None,
    ) -> None:
        """Initialize NS-SE engine.

        Args:
            akg: Algorithmic Knowledge Graph (constructed offline by LLM)
            fitness_function: Function(algorithm, instance) -> fitness
            problem_instance: Problem instance to solve
            problem_type: Type of problem (tsp, jssp, vrp)
            config: Engine configuration
            trajectory_db_path: Path for trajectory database
        """
        self.akg = akg
        self.fitness_function = fitness_function
        self.problem_instance = problem_instance
        self.problem_type = problem_type
        self.config = config or EngineConfig()

        # Get problem size
        self.problem_size = getattr(problem_instance, "dimension", 0)

        # Initialize population manager
        self.population_manager = PopulationManager(
            akg=akg,
            population_size=self.config.population_size,
            min_operators=self.config.min_operators,
            max_operators=self.config.max_operators,
        )

        # Initialize ACO/MMAS selector for operator selection
        aco_config = self.config.aco_config or ACOConfig(
            max_steps=self.config.max_operators,
            enable_conditions=True,  # Enable Level 3 conditional transitions
        )
        self.aco_selector = ACOSelector(akg, aco_config)

        # Trajectory storage
        self.trajectory_lookup = TrajectoryLookup(trajectory_db_path)

        # State
        self.population: Population | None = None
        self.stats = EvolutionStats()
        self.best_algorithm: Algorithm | None = None

        # Search state tracking (for conditions and pheromone updates)
        self._generations_without_improvement = 0
        self._recent_improvements: list[bool] = []  # Track last N improvements

    def run(self) -> Algorithm:
        """Run the evolution loop.

        Returns:
            Best algorithm found
        """
        start_time = time.time()

        logger.info(f"Starting NS-SE evolution on {self.problem_type} "
                   f"(size={self.problem_size}) - MMAS only, no LLM at runtime")

        # Initialize population using ACO
        self.population = self._initialize_population_with_aco()
        logger.info(f"Initialized population with {self.population.size} algorithms")

        # Evaluate initial population
        self._evaluate_population()
        self._update_best()

        logger.info(f"Initial best fitness: {self.stats.best_fitness:.2f}")

        # Evolution loop
        generations_without_improvement = 0
        last_best = self.stats.best_fitness

        while not self._should_stop(generations_without_improvement):
            self.stats.generations += 1

            if self.config.verbose:
                logger.info(f"Generation {self.stats.generations}")

            # Update execution context for Level 3 conditions
            context = self.build_execution_context()
            self.aco_selector.set_execution_context(context)

            # Generate offspring using MMAS
            offspring = self._generate_offspring_with_aco()

            # Evaluate offspring
            for child in offspring:
                if self.stats.evaluations >= self.config.max_evaluations:
                    break
                self._evaluate_algorithm(child)

            # Replace population
            self.population = self.population_manager.elitist_replacement(
                self.population, offspring, self.config.elite_count
            )

            # Update best
            prev_best = self.stats.best_fitness
            self._update_best()

            # Update pheromones based on best solution
            if self.best_algorithm:
                self.aco_selector.update_pheromones_for_path(
                    self.best_algorithm.operators,
                    self.best_algorithm.fitness or float("inf")
                )
                self.stats.pheromone_updates += 1

            # Check convergence
            improved = self.stats.best_fitness < prev_best * (1 - self.config.convergence_threshold)
            self._recent_improvements.append(improved)
            if len(self._recent_improvements) > 10:
                self._recent_improvements = self._recent_improvements[-10:]

            if improved:
                generations_without_improvement = 0
                self._generations_without_improvement = 0
                last_best = self.stats.best_fitness
            else:
                generations_without_improvement += 1
                self._generations_without_improvement = generations_without_improvement

            # Record history
            self.stats.best_fitness_history.append(self.stats.best_fitness)
            avg = self.population.average_fitness
            if avg:
                self.stats.avg_fitness_history.append(avg)

            if self.config.verbose:
                avg_str = f"{avg:.2f}" if avg else "N/A"
                logger.info(f"  Best: {self.stats.best_fitness:.2f}, "
                           f"Avg: {avg_str}, "
                           f"Evals: {self.stats.evaluations}")

        self.stats.wall_time_seconds = time.time() - start_time

        # Record best trajectory
        if self.best_algorithm:
            self._record_trajectory(self.best_algorithm)

        logger.info(f"Evolution complete: {self.stats.generations} generations, "
                   f"{self.stats.evaluations} evaluations, "
                   f"{self.stats.aco_solutions_constructed} ACO solutions, "
                   f"best fitness: {self.stats.best_fitness:.2f}")

        return self.best_algorithm

    def _should_stop(self, generations_without_improvement: int) -> bool:
        """Check stopping criteria.

        Args:
            generations_without_improvement: Generations without improvement

        Returns:
            True if should stop
        """
        # Max generations
        if self.stats.generations >= self.config.max_generations:
            logger.info("Stopping: max generations reached")
            return True

        # Max evaluations
        if self.stats.evaluations >= self.config.max_evaluations:
            logger.info("Stopping: max evaluations reached")
            return True

        # Convergence
        if generations_without_improvement >= self.config.convergence_generations:
            logger.info("Stopping: convergence reached")
            return True

        return False

    def _evaluate_population(self) -> None:
        """Evaluate all algorithms in population."""
        for algorithm in self.population.algorithms:
            if algorithm.fitness is None:
                self._evaluate_algorithm(algorithm)

    def _evaluate_algorithm(self, algorithm: Algorithm) -> None:
        """Evaluate a single algorithm.

        Args:
            algorithm: Algorithm to evaluate
        """
        try:
            fitness = self.fitness_function(algorithm, self.problem_instance)
            algorithm.fitness = fitness
            self.stats.evaluations += 1

        except Exception as e:
            logger.warning(f"Evaluation failed for {algorithm.id}: {e}")
            algorithm.fitness = float("inf")

    def _update_best(self) -> None:
        """Update best algorithm tracking."""
        best = self.population.best_algorithm
        if best and (self.best_algorithm is None or best.fitness < self.best_algorithm.fitness):
            self.best_algorithm = best
            self.stats.best_fitness = best.fitness

    def _initialize_population_with_aco(self) -> Population:
        """Initialize population using ACO to construct operator sequences.

        Returns:
            Initial population
        """
        algorithms = []

        for i in range(self.config.population_size):
            # Use ACO to construct an operator sequence
            ant = self.aco_selector.construct_solution()
            self.stats.aco_solutions_constructed += 1

            algorithm = Algorithm(
                id=self.population_manager._generate_id(),
                operators=ant.path,
                mutation_history=["aco_init"],
            )
            algorithms.append(algorithm)

        return Population(algorithms=algorithms)

    def _generate_offspring_with_aco(self) -> list[Algorithm]:
        """Generate offspring using ACO/MMAS traversal.

        Each offspring is created by constructing a new solution using
        MMAS, which uses pheromones + edge weights + conditions.

        Returns:
            List of offspring algorithms
        """
        offspring = []
        n_offspring = self.config.population_size - self.config.elite_count

        for _ in range(n_offspring):
            # Use ACO to construct a new operator sequence
            ant = self.aco_selector.construct_solution()
            self.stats.aco_solutions_constructed += 1

            algorithm = Algorithm(
                id=self.population_manager._generate_id(),
                operators=ant.path,
                mutation_history=["aco_construct"],
            )
            offspring.append(algorithm)

        return offspring

    def _compute_population_diversity(self) -> float:
        """Compute population diversity based on unique operator sequences.

        Returns:
            Diversity score between 0 and 1
        """
        if not self.population or self.population.size < 2:
            return 0.5

        # Count unique operator sequences
        unique_sequences = set()
        for alg in self.population.algorithms:
            unique_sequences.add(tuple(alg.operators))

        # Diversity = unique / total
        return len(unique_sequences) / self.population.size

    def build_execution_context(self) -> ExecutionContext:
        """Build execution context for Level 3 conditional transitions.

        The ExecutionContext provides runtime metrics used to evaluate
        edge conditions during ACO traversal.

        Returns:
            ExecutionContext with current search state
        """
        # Calculate improvement rate from recent improvements
        if self._recent_improvements:
            improved_count = sum(1 for improved in self._recent_improvements[-5:] if improved)
            improvement_rate = improved_count / min(5, len(self._recent_improvements))
        else:
            improvement_rate = 0.0

        # Count consecutive local search operations in best algorithm
        consecutive_ls = 0
        if self.best_algorithm and self.best_algorithm.operators:
            for op_id in reversed(self.best_algorithm.operators):
                node = self.akg.get_node(op_id)
                if node and hasattr(node, 'category'):
                    if node.category == OperatorCategory.LOCAL_SEARCH:
                        consecutive_ls += 1
                    else:
                        break

        # Get current fitness (from best algorithm or population best)
        current_fitness = float("inf")
        if self.best_algorithm and self.best_algorithm.fitness is not None:
            current_fitness = self.best_algorithm.fitness
        elif self.population and self.population.best_algorithm:
            current_fitness = self.population.best_algorithm.fitness or float("inf")

        return ExecutionContext(
            generations_without_improvement=self._generations_without_improvement,
            current_fitness=current_fitness,
            best_fitness=self.stats.best_fitness,
            population_diversity=self._compute_population_diversity(),
            evaluations_used=self.stats.evaluations,
            max_evaluations=self.config.max_evaluations,
            consecutive_local_search=consecutive_ls,
            recent_improvement_rate=improvement_rate,
            current_operators=self.best_algorithm.operators if self.best_algorithm else [],
        )

    def _record_trajectory(self, algorithm: Algorithm) -> None:
        """Record successful trajectory to database.

        Args:
            algorithm: Algorithm to record
        """
        if self.config.log_all_trajectories:
            trajectory = Trajectory(
                id=algorithm.id,
                operators=algorithm.operators,
                problem_type=self.problem_type,
                problem_size=self.problem_size,
                fitness=algorithm.fitness or 0.0,
                metadata=algorithm.metadata,
            )

            self.trajectory_lookup.add_trajectory(trajectory)
            self.akg.add_trajectory(trajectory)

    def get_stats(self) -> EvolutionStats:
        """Get evolution statistics.

        Returns:
            Statistics object
        """
        return self.stats

    def get_best_algorithm(self) -> Algorithm | None:
        """Get best algorithm found.

        Returns:
            Best algorithm or None
        """
        return self.best_algorithm

    def close(self) -> None:
        """Close resources."""
        self.trajectory_lookup.close()
