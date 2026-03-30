"""Algorithm representation and population management."""

import random
from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import AlgorithmicKnowledgeGraph, Trajectory
from src.geakg.nodes import OperatorCategory, OperatorNode


class Algorithm(BaseModel):
    """Representation of an algorithm as a sequence of operators.

    An algorithm in NS-SE is represented as an ordered sequence of
    operator IDs from the AKG, forming a pipeline that can be executed
    on any problem instance.
    """

    id: str
    operators: list[str]  # List of operator node IDs
    fitness: float | None = None
    generation: int = 0
    parent_id: str | None = None
    mutation_history: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of operators."""
        return len(self.operators)

    def __repr__(self) -> str:
        """String representation."""
        ops = " -> ".join(self.operators[:3])
        if len(self.operators) > 3:
            ops += " -> ..."
        fitness_str = f"{self.fitness:.2f}" if self.fitness else "N/A"
        return f"Algorithm(id={self.id}, ops=[{ops}], fitness={fitness_str})"

    def to_trajectory(self, problem_type: str, problem_size: int) -> Trajectory:
        """Convert algorithm to trajectory for AKG storage.

        Args:
            problem_type: Problem domain (tsp, jssp, vrp)
            problem_size: Size of problem instance

        Returns:
            Trajectory object
        """
        return Trajectory(
            id=self.id,
            operators=self.operators,
            problem_type=problem_type,
            problem_size=problem_size,
            fitness=self.fitness or 0.0,
            metadata=self.metadata,
        )


class Population(BaseModel):
    """Population of algorithms for evolutionary search."""

    algorithms: list[Algorithm] = Field(default_factory=list)
    generation: int = 0
    best_fitness_history: list[float] = Field(default_factory=list)
    avg_fitness_history: list[float] = Field(default_factory=list)

    @property
    def size(self) -> int:
        """Current population size."""
        return len(self.algorithms)

    @property
    def best_algorithm(self) -> Algorithm | None:
        """Get best algorithm by fitness (lower is better)."""
        if not self.algorithms:
            return None

        evaluated = [a for a in self.algorithms if a.fitness is not None]
        if not evaluated:
            return None

        return min(evaluated, key=lambda a: a.fitness)

    @property
    def average_fitness(self) -> float | None:
        """Get average fitness of population."""
        evaluated = [a for a in self.algorithms if a.fitness is not None]
        if not evaluated:
            return None
        return sum(a.fitness for a in evaluated) / len(evaluated)

    def add_algorithm(self, algorithm: Algorithm) -> None:
        """Add algorithm to population.

        Args:
            algorithm: Algorithm to add
        """
        algorithm.generation = self.generation
        self.algorithms.append(algorithm)

    def update_history(self) -> None:
        """Update fitness history for tracking convergence."""
        best = self.best_algorithm
        if best and best.fitness is not None:
            self.best_fitness_history.append(best.fitness)

        avg = self.average_fitness
        if avg is not None:
            self.avg_fitness_history.append(avg)

    def next_generation(self) -> None:
        """Increment generation counter."""
        self.generation += 1
        self.update_history()


class PopulationManager:
    """Manages population initialization and evolution."""

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        population_size: int = 20,
        min_operators: int = 2,
        max_operators: int = 6,
    ) -> None:
        """Initialize population manager.

        Args:
            akg: Algorithmic Knowledge Graph
            population_size: Target population size
            min_operators: Minimum operators per algorithm
            max_operators: Maximum operators per algorithm
        """
        self.akg = akg
        self.population_size = population_size
        self.min_operators = min_operators
        self.max_operators = max_operators
        self._id_counter = 0

    def _generate_id(self) -> str:
        """Generate unique algorithm ID."""
        self._id_counter += 1
        return f"alg_{self._id_counter:04d}"

    def initialize_population(self) -> Population:
        """Create initial random population.

        Returns:
            New population with random algorithms
        """
        population = Population()

        for _ in range(self.population_size):
            algorithm = self.random_algorithm()
            population.add_algorithm(algorithm)

        return population

    def random_algorithm(self) -> Algorithm:
        """Generate a random valid algorithm.

        Returns:
            Random algorithm following AKG constraints
        """
        n_operators = random.randint(self.min_operators, self.max_operators)
        operators = []

        # Start with a construction operator
        construction_ops = self.akg.get_operators_by_category(OperatorCategory.CONSTRUCTION)
        if construction_ops:
            first_op = random.choice(construction_ops)
            operators.append(first_op.id)

        # Add more operators following valid transitions
        while len(operators) < n_operators:
            current_op = operators[-1]
            valid_next = self.akg.get_valid_transitions(current_op)

            if not valid_next:
                break

            # Weight selection by edge weights
            weights = []
            for next_op in valid_next:
                edge = self.akg.edges.get((current_op, next_op))
                weights.append(edge.weight if edge else 0.5)

            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                next_op = random.choices(valid_next, weights=weights)[0]
            else:
                next_op = random.choice(valid_next)

            operators.append(next_op)

        return Algorithm(
            id=self._generate_id(),
            operators=operators,
        )

    def mutate_algorithm(
        self,
        algorithm: Algorithm,
        mutation_type: str = "replace",
    ) -> Algorithm:
        """Create mutated copy of algorithm.

        Args:
            algorithm: Algorithm to mutate
            mutation_type: Type of mutation (replace, insert, delete, extend)

        Returns:
            New mutated algorithm
        """
        new_operators = algorithm.operators.copy()

        if mutation_type == "replace" and len(new_operators) > 1:
            # Replace a random operator (not the first construction one)
            idx = random.randint(1, len(new_operators) - 1)
            prev_op = new_operators[idx - 1]
            valid_next = self.akg.get_valid_transitions(prev_op)

            if valid_next:
                new_op = random.choice(valid_next)
                new_operators[idx] = new_op

        elif mutation_type == "insert" and len(new_operators) < self.max_operators:
            # Insert a new operator
            idx = random.randint(1, len(new_operators))
            prev_op = new_operators[idx - 1]
            valid_next = self.akg.get_valid_transitions(prev_op)

            if valid_next:
                new_op = random.choice(valid_next)
                new_operators.insert(idx, new_op)

        elif mutation_type == "delete" and len(new_operators) > self.min_operators:
            # Delete a random operator (not the first construction one)
            idx = random.randint(1, len(new_operators) - 1)
            new_operators.pop(idx)

        elif mutation_type == "extend":
            # Add operator at the end
            if len(new_operators) < self.max_operators:
                last_op = new_operators[-1]
                valid_next = self.akg.get_valid_transitions(last_op)

                if valid_next:
                    new_op = random.choice(valid_next)
                    new_operators.append(new_op)

        child = Algorithm(
            id=self._generate_id(),
            operators=new_operators,
            parent_id=algorithm.id,
            mutation_history=algorithm.mutation_history + [mutation_type],
        )

        return child

    def tournament_select(
        self,
        population: Population,
        tournament_size: int = 3,
    ) -> Algorithm:
        """Select algorithm using tournament selection.

        Args:
            population: Current population
            tournament_size: Number of algorithms in tournament

        Returns:
            Selected algorithm
        """
        # Filter to evaluated algorithms
        evaluated = [a for a in population.algorithms if a.fitness is not None]
        if not evaluated:
            return random.choice(population.algorithms)

        # Select tournament participants
        participants = random.sample(
            evaluated, min(tournament_size, len(evaluated))
        )

        # Return best (lowest fitness for minimization)
        return min(participants, key=lambda a: a.fitness)

    def elitist_replacement(
        self,
        population: Population,
        offspring: list[Algorithm],
        elite_count: int = 2,
    ) -> Population:
        """Replace population with elitist strategy.

        Args:
            population: Current population
            offspring: New offspring algorithms
            elite_count: Number of elite to preserve

        Returns:
            New population
        """
        # Get elite from current population
        evaluated = [a for a in population.algorithms if a.fitness is not None]
        evaluated.sort(key=lambda a: a.fitness)
        elite = evaluated[:elite_count]

        # Combine elite with best offspring
        all_algorithms = elite + offspring
        all_algorithms.sort(key=lambda a: a.fitness if a.fitness else float("inf"))

        # Create new population
        new_pop = Population(generation=population.generation + 1)
        for alg in all_algorithms[: self.population_size]:
            new_pop.algorithms.append(alg)

        new_pop.update_history()
        return new_pop
