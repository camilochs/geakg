"""Meta-heuristic operators for TSP.

These operators implement steps of meta-heuristic algorithms.
They can be composed to form complete meta-heuristics.
6 operators total.
"""

import math
import random
from typing import TypeAlias, Any

from src.operators.base import (
    Tour,
    DistanceMatrix,
    calculate_tour_cost,
)
from src.operators.local_search import two_opt, swap_operator
from src.operators.perturbation import double_bridge


class SimulatedAnnealingState:
    """State for Simulated Annealing."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.1
    ):
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations = 0

    def cool(self):
        """Apply cooling schedule."""
        self.temperature = max(
            self.min_temp,
            self.temperature * self.cooling_rate
        )
        self.iterations += 1

    def accept_probability(self, delta: float) -> float:
        """Calculate acceptance probability for worse solution."""
        if delta <= 0:
            return 1.0
        if self.temperature <= 0:
            return 0.0
        return math.exp(-delta / self.temperature)


class TabuList:
    """Tabu list for Tabu Search."""

    def __init__(self, tenure: int = 7):
        self.tenure = tenure
        self.tabu_dict: dict[tuple[int, int], int] = {}
        self.iteration = 0

    def add(self, move: tuple[int, int]):
        """Add a move to the tabu list."""
        self.tabu_dict[move] = self.iteration + self.tenure
        # Also add reverse move
        self.tabu_dict[(move[1], move[0])] = self.iteration + self.tenure

    def is_tabu(self, move: tuple[int, int]) -> bool:
        """Check if a move is tabu."""
        if move in self.tabu_dict:
            return self.tabu_dict[move] > self.iteration
        return False

    def step(self):
        """Increment iteration and clean expired entries."""
        self.iteration += 1
        # Clean expired entries periodically
        if self.iteration % 100 == 0:
            self.tabu_dict = {
                k: v for k, v in self.tabu_dict.items()
                if v > self.iteration
            }


def simulated_annealing_step(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    sa_state: SimulatedAnnealingState | None = None,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    neighbor_operator: str = "two_opt_move",
    **kwargs
) -> tuple[Tour, SimulatedAnnealingState]:
    """Single step of Simulated Annealing.

    Generates a neighbor solution and accepts/rejects based on
    Metropolis criterion.

    Time complexity: O(n) for neighbor generation

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        sa_state: SA state (temperature, etc). Created if None.
        initial_temp: Initial temperature
        cooling_rate: Cooling rate
        neighbor_operator: Type of move ("two_opt_move", "swap", "insert")

    Returns:
        Tuple of (new_tour, sa_state)
    """
    n = len(tour)
    if n < 3:
        if sa_state is None:
            sa_state = SimulatedAnnealingState(initial_temp, cooling_rate)
        return tour.copy(), sa_state

    # Initialize state if needed
    if sa_state is None:
        sa_state = SimulatedAnnealingState(initial_temp, cooling_rate)

    current_cost = calculate_tour_cost(tour, distance_matrix)
    new_tour = tour.copy()

    # Generate neighbor
    if neighbor_operator == "swap":
        # Swap two random cities
        i, j = random.sample(range(n), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

    elif neighbor_operator == "insert":
        # Remove and reinsert a city
        i = random.randint(0, n - 1)
        city = new_tour.pop(i)
        j = random.randint(0, len(new_tour))
        new_tour.insert(j, city)

    else:  # two_opt_move (default)
        # Single 2-opt move
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        new_tour[i:j + 1] = new_tour[i:j + 1][::-1]

    new_cost = calculate_tour_cost(new_tour, distance_matrix)
    delta = new_cost - current_cost

    # Metropolis criterion
    if delta <= 0 or random.random() < sa_state.accept_probability(delta):
        result_tour = new_tour
    else:
        result_tour = tour.copy()

    # Cool down
    sa_state.cool()

    return result_tour, sa_state


def tabu_search_step(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    tabu_list: TabuList | None = None,
    tabu_tenure: int = 7,
    aspiration: bool = True,
    best_cost: float | None = None,
    **kwargs
) -> tuple[Tour, TabuList, float]:
    """Single step of Tabu Search.

    Explores neighborhood avoiding tabu moves unless aspiration
    criterion is met.

    Time complexity: O(n²) for full neighborhood scan

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        tabu_list: Tabu list. Created if None.
        tabu_tenure: Tabu tenure (how long moves stay tabu)
        aspiration: Enable aspiration criterion
        best_cost: Best known cost (for aspiration)

    Returns:
        Tuple of (new_tour, tabu_list, best_cost)
    """
    n = len(tour)
    if n < 4:
        if tabu_list is None:
            tabu_list = TabuList(tabu_tenure)
        current_cost = calculate_tour_cost(tour, distance_matrix)
        return tour.copy(), tabu_list, best_cost or current_cost

    # Initialize if needed
    if tabu_list is None:
        tabu_list = TabuList(tabu_tenure)

    current_cost = calculate_tour_cost(tour, distance_matrix)
    if best_cost is None:
        best_cost = current_cost

    # Find best non-tabu move (or tabu move with aspiration)
    best_move = None
    best_move_cost = float("inf")
    best_move_i, best_move_j = -1, -1

    # Explore 2-opt neighborhood
    for i in range(n - 1):
        for j in range(i + 2, n):
            if j == n - 1 and i == 0:
                continue

            # Check if move is tabu
            move = (tour[i], tour[j])
            is_tabu = tabu_list.is_tabu(move)

            # Calculate move cost
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]

            old_cost = distance_matrix[a][b] + distance_matrix[c][d]
            new_cost = distance_matrix[a][c] + distance_matrix[b][d]
            delta = new_cost - old_cost

            move_cost = current_cost + delta

            # Accept if not tabu, or if aspiration criterion met
            if not is_tabu or (aspiration and move_cost < best_cost):
                if move_cost < best_move_cost:
                    best_move_cost = move_cost
                    best_move_i, best_move_j = i, j
                    best_move = move

    # Apply best move
    if best_move is not None:
        new_tour = tour.copy()
        new_tour[best_move_i + 1:best_move_j + 1] = \
            new_tour[best_move_i + 1:best_move_j + 1][::-1]

        # Add move to tabu list
        tabu_list.add(best_move)

        # Update best cost
        if best_move_cost < best_cost:
            best_cost = best_move_cost

        result_tour = new_tour
    else:
        result_tour = tour.copy()

    tabu_list.step()

    return result_tour, tabu_list, best_cost


def genetic_crossover(
    tour1: Tour,
    tour2: Tour,
    distance_matrix: DistanceMatrix,
    crossover_type: str = "order",
    **kwargs
) -> Tour:
    """Genetic crossover between two parent tours.

    Combines two parent tours to create offspring.

    Time complexity: O(n)

    Args:
        tour1: First parent tour
        tour2: Second parent tour
        distance_matrix: Distance matrix
        crossover_type: Type of crossover ("order", "pmx", "cycle")

    Returns:
        Child tour
    """
    n = len(tour1)
    if n != len(tour2):
        return tour1.copy()

    if n < 4:
        return tour1.copy()

    if crossover_type == "pmx":
        # Partially Mapped Crossover (PMX)
        # Select two crossover points
        cp1 = random.randint(0, n - 2)
        cp2 = random.randint(cp1 + 1, n - 1)

        # Initialize child with -1
        child = [-1] * n

        # Copy segment from parent 1
        child[cp1:cp2 + 1] = tour1[cp1:cp2 + 1]

        # Create mapping
        mapping = {}
        for i in range(cp1, cp2 + 1):
            mapping[tour1[i]] = tour2[i]

        # Fill rest from parent 2
        for i in range(n):
            if cp1 <= i <= cp2:
                continue

            city = tour2[i]
            while city in child:
                city = mapping.get(city, city)
                if city == tour2[i]:
                    break

            if city not in child:
                child[i] = city

        # Fill any remaining with unused cities
        used = set(c for c in child if c != -1)
        unused = [c for c in range(n) if c not in used]
        random.shuffle(unused)

        for i in range(n):
            if child[i] == -1:
                child[i] = unused.pop()

        return child

    elif crossover_type == "cycle":
        # Cycle Crossover (CX)
        child = [-1] * n
        pos2 = {city: i for i, city in enumerate(tour2)}

        cycle = 0
        while -1 in child:
            # Find first unfilled position
            start = child.index(-1)

            # Build cycle
            current = start
            while True:
                if cycle % 2 == 0:
                    child[current] = tour1[current]
                else:
                    child[current] = tour2[current]

                # Find position in tour2 of city we just placed from tour1
                next_city = tour2[current]
                current = tour1.index(next_city)

                if current == start:
                    break

            cycle += 1

        return child

    else:  # order (default) - Order Crossover (OX)
        # Select two crossover points
        cp1 = random.randint(0, n - 2)
        cp2 = random.randint(cp1 + 1, n - 1)

        # Initialize child
        child = [-1] * n

        # Copy segment from parent 1
        child[cp1:cp2 + 1] = tour1[cp1:cp2 + 1]

        # Fill rest from parent 2 in order
        in_child = set(child[cp1:cp2 + 1])
        parent2_order = [c for c in tour2 if c not in in_child]

        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = parent2_order[idx]
                idx += 1

        return child


class AntColonyState:
    """State for Ant Colony Optimization."""

    def __init__(
        self,
        n_cities: int,
        initial_pheromone: float = 1.0,
        evaporation_rate: float = 0.1,
        alpha: float = 1.0,
        beta: float = 2.0,
        q: float = 1.0
    ):
        self.n = n_cities
        self.pheromone = [[initial_pheromone] * n_cities for _ in range(n_cities)]
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.q = q          # Pheromone deposit factor
        self.best_tour: Tour | None = None
        self.best_cost = float("inf")


def ant_colony_update(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    aco_state: AntColonyState | None = None,
    evaporation_rate: float = 0.1,
    q: float = 1.0,
    **kwargs
) -> tuple[Tour, AntColonyState]:
    """Ant Colony Optimization update step.

    Updates pheromone trails based on tour quality and
    constructs new solution probabilistically.

    Time complexity: O(n²)

    Args:
        tour: Current tour (used for pheromone update)
        distance_matrix: Distance matrix
        aco_state: ACO state. Created if None.
        evaporation_rate: Pheromone evaporation rate
        q: Pheromone deposit factor

    Returns:
        Tuple of (new_tour, aco_state)
    """
    n = len(tour)
    if n < 3:
        if aco_state is None:
            aco_state = AntColonyState(n, evaporation_rate=evaporation_rate, q=q)
        return tour.copy(), aco_state

    # Initialize state if needed
    if aco_state is None:
        aco_state = AntColonyState(n, evaporation_rate=evaporation_rate, q=q)

    tour_cost = calculate_tour_cost(tour, distance_matrix)

    # Update best
    if tour_cost < aco_state.best_cost:
        aco_state.best_cost = tour_cost
        aco_state.best_tour = tour.copy()

    # Evaporate pheromones
    for i in range(n):
        for j in range(n):
            aco_state.pheromone[i][j] *= (1 - aco_state.evaporation_rate)
            aco_state.pheromone[i][j] = max(0.01, aco_state.pheromone[i][j])

    # Deposit pheromone on tour edges
    deposit = aco_state.q / tour_cost
    for i in range(n):
        j = (i + 1) % n
        a, b = tour[i], tour[j]
        aco_state.pheromone[a][b] += deposit
        aco_state.pheromone[b][a] += deposit

    # Construct new tour probabilistically
    new_tour = [random.randint(0, n - 1)]
    visited = set(new_tour)

    while len(new_tour) < n:
        current = new_tour[-1]
        probabilities = []

        for city in range(n):
            if city in visited:
                probabilities.append(0)
            else:
                pheromone = aco_state.pheromone[current][city] ** aco_state.alpha
                visibility = (1.0 / max(distance_matrix[current][city], 0.001)) ** aco_state.beta
                probabilities.append(pheromone * visibility)

        total = sum(probabilities)
        if total == 0:
            # Random selection
            unvisited = [c for c in range(n) if c not in visited]
            next_city = random.choice(unvisited)
        else:
            # Roulette wheel selection
            r = random.random() * total
            cumsum = 0
            next_city = 0
            for city, prob in enumerate(probabilities):
                cumsum += prob
                if cumsum >= r:
                    next_city = city
                    break

        new_tour.append(next_city)
        visited.add(next_city)

    return new_tour, aco_state


class ParticleSwarmState:
    """State for Particle Swarm Optimization (discrete version)."""

    def __init__(
        self,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5
    ):
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.personal_best: Tour | None = None
        self.personal_best_cost = float("inf")
        self.global_best: Tour | None = None
        self.global_best_cost = float("inf")
        self.velocity: list[tuple[int, int]] = []  # List of swap operations


def particle_swarm_update(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    pso_state: ParticleSwarmState | None = None,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    **kwargs
) -> tuple[Tour, ParticleSwarmState]:
    """Particle Swarm Optimization update for discrete TSP.

    Uses swap sequence representation for velocity.

    Time complexity: O(n²)

    Args:
        tour: Current tour (particle position)
        distance_matrix: Distance matrix
        pso_state: PSO state. Created if None.
        inertia: Inertia weight
        cognitive: Cognitive coefficient
        social: Social coefficient

    Returns:
        Tuple of (new_tour, pso_state)
    """
    n = len(tour)
    if n < 3:
        if pso_state is None:
            pso_state = ParticleSwarmState(inertia, cognitive, social)
        return tour.copy(), pso_state

    # Initialize state if needed
    if pso_state is None:
        pso_state = ParticleSwarmState(inertia, cognitive, social)

    current_cost = calculate_tour_cost(tour, distance_matrix)

    # Update personal best
    if current_cost < pso_state.personal_best_cost:
        pso_state.personal_best_cost = current_cost
        pso_state.personal_best = tour.copy()

    # Update global best
    if current_cost < pso_state.global_best_cost:
        pso_state.global_best_cost = current_cost
        pso_state.global_best = tour.copy()

    new_tour = tour.copy()

    # Apply inertia (keep some of previous velocity/swaps)
    if pso_state.velocity and random.random() < pso_state.inertia:
        # Apply some swaps from velocity
        n_swaps = max(1, int(len(pso_state.velocity) * pso_state.inertia))
        for i, j in random.sample(pso_state.velocity, min(n_swaps, len(pso_state.velocity))):
            if i < n and j < n:
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

    # Cognitive component (move toward personal best)
    if pso_state.personal_best and random.random() < pso_state.cognitive / 3:
        # Find swaps needed to get closer to personal best
        pb = pso_state.personal_best
        for i in range(n):
            if new_tour[i] != pb[i]:
                # Find where pb[i] is in new_tour
                j = new_tour.index(pb[i])
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                pso_state.velocity.append((i, j))
                break

    # Social component (move toward global best)
    if pso_state.global_best and random.random() < pso_state.social / 3:
        gb = pso_state.global_best
        for i in range(n):
            if new_tour[i] != gb[i]:
                j = new_tour.index(gb[i])
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                pso_state.velocity.append((i, j))
                break

    # Limit velocity size
    if len(pso_state.velocity) > n:
        pso_state.velocity = pso_state.velocity[-n:]

    return new_tour, pso_state


def iterated_local_search_step(
    tour: Tour,
    distance_matrix: DistanceMatrix,
    perturbation_op: str = "double_bridge",
    local_search_op: str = "two_opt",
    acceptance: str = "improving",
    best_tour: Tour | None = None,
    best_cost: float | None = None,
    **kwargs
) -> tuple[Tour, Tour | None, float | None]:
    """Single step of Iterated Local Search.

    ILS cycle:
    1. Perturb current solution
    2. Apply local search
    3. Accept/reject based on criterion

    Time complexity: Depends on local search used

    Args:
        tour: Current tour
        distance_matrix: Distance matrix
        perturbation_op: Perturbation operator to use
        local_search_op: Local search operator to use
        acceptance: Acceptance criterion ("improving", "always", "sa")
        best_tour: Best tour found so far
        best_cost: Best cost found so far

    Returns:
        Tuple of (new_tour, best_tour, best_cost)
    """
    n = len(tour)
    if n < 4:
        current_cost = calculate_tour_cost(tour, distance_matrix) if n > 0 else 0
        return tour.copy(), tour.copy(), current_cost

    current_cost = calculate_tour_cost(tour, distance_matrix)

    if best_tour is None:
        best_tour = tour.copy()
        best_cost = current_cost
    elif best_cost is None:
        best_cost = calculate_tour_cost(best_tour, distance_matrix)

    # Step 1: Perturb
    if perturbation_op == "double_bridge":
        perturbed = double_bridge(tour, distance_matrix)
    else:
        # Random perturbation
        perturbed = tour.copy()
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        perturbed[i:j + 1] = perturbed[i:j + 1][::-1]

    # Step 2: Local search
    if local_search_op == "two_opt":
        improved = two_opt(perturbed, distance_matrix, max_iterations=500)
    else:
        improved = swap_operator(perturbed, distance_matrix, max_iterations=100)

    improved_cost = calculate_tour_cost(improved, distance_matrix)

    # Step 3: Acceptance
    if acceptance == "always":
        new_tour = improved
    elif acceptance == "sa":
        # Simulated annealing-like acceptance
        delta = improved_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / 10):
            new_tour = improved
        else:
            new_tour = tour.copy()
    else:  # improving (default)
        if improved_cost < current_cost:
            new_tour = improved
        else:
            new_tour = tour.copy()

    # Update best
    new_cost = calculate_tour_cost(new_tour, distance_matrix)
    if new_cost < best_cost:
        best_tour = new_tour.copy()
        best_cost = new_cost

    return new_tour, best_tour, best_cost
