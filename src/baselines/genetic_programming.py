"""Genetic Programming baseline using DEAP.

Standard GP baseline for comparison with NS-SE.
Uses same budget (1000 evaluations) for fair comparison.

This is a FULLY ENHANCED GP baseline that includes ALL 30 operators
that NS-SE has access to, ensuring a completely fair comparison:

Construction (10):
1. nearest_neighbor, 2. farthest_insertion, 3. cheapest_insertion,
4. random_insertion, 5. savings_heuristic, 6. christofides_construction,
7. nearest_addition, 8. convex_hull_start, 9. cluster_first, 10. sweep_algorithm

Local Search (8):
11. two_opt, 12. three_opt, 13. or_opt, 14. swap,
15. insert, 16. invert, 17. lin_kernighan, 18. variable_neighborhood

Perturbation (6):
19. double_bridge, 20. segment_shuffle, 21. guided_mutation,
22. ruin_recreate, 23. large_neighborhood_search, 24. adaptive_mutation

Meta-heuristic (6):
25. simulated_annealing, 26. tabu_search, 27. genetic_crossover,
28. ant_colony, 29. particle_swarm, 30. iterated_local_search

This provides a 100% fair comparison for IEEE TEVC.
"""

import math
import operator
import random
import time
from typing import Any

from deap import base, creator, gp, tools
from pydantic import BaseModel, Field


class GPResult(BaseModel):
    """Result from GP run."""

    best_fitness: float
    best_individual: str = ""
    evaluations: int = 0
    generations: int = 0
    wall_time_seconds: float = 0.0
    fitness_history: list[float] = Field(default_factory=list)


class TSPGeneticProgramming:
    """Genetic Programming for TSP using DEAP.

    Evolves programs that construct TSP solutions by combining
    ALL 30 algorithmic primitives available to NS-SE.

    This ensures a completely fair comparison where GP has access
    to the exact same building blocks as NS-SE.
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
        """Initialize GP baseline."""
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
        distance_matrix: list[list[float]],
    ) -> GPResult:
        """Run GP optimization for TSP."""
        start_time = time.time()
        n = len(distance_matrix)
        self._eval_count = 0
        self._fitness_history = []

        # Define primitive set for TSP construction
        pset = self._create_primitive_set(n)

        # Create fitness and individual types
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        # Create toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=self.max_tree_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Evaluation function
        def evaluate(individual: Any) -> tuple[float]:
            if self._eval_count >= self.budget:
                return (float("inf"),)

            try:
                func = toolbox.compile(expr=individual)
                tour = func(distance_matrix)

                # Validate tour
                if not self._validate_tour(tour, n):
                    return (float("inf"),)

                # Calculate cost
                cost = self._calculate_tour_cost(tour, distance_matrix)
                self._eval_count += 1
                self._fitness_history.append(cost)
                return (cost,)

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

        return GPResult(
            best_fitness=best_fitness,
            best_individual=str(best) if best else "",
            evaluations=self._eval_count,
            generations=generation + 1,
            wall_time_seconds=wall_time,
            fitness_history=self._fitness_history,
        )

    def _create_primitive_set(self, n: int) -> gp.PrimitiveSet:
        """Create primitive set with ALL 30 TSP operators.

        Args:
            n: Number of cities

        Returns:
            DEAP primitive set with 30 primitives
        """
        pset = gp.PrimitiveSet("MAIN", 1)  # Takes distance matrix

        # =====================================================================
        # CONSTRUCTION PRIMITIVES (10)
        # =====================================================================

        def nearest_neighbor(matrix: list[list[float]]) -> list[int]:
            """Greedy nearest neighbor construction."""
            n = len(matrix)
            start = random.randint(0, n - 1)
            tour = [start]
            visited = {start}

            while len(tour) < n:
                current = tour[-1]
                best_next = None
                best_dist = float("inf")

                for city in range(n):
                    if city not in visited and matrix[current][city] < best_dist:
                        best_dist = matrix[current][city]
                        best_next = city

                if best_next is not None:
                    tour.append(best_next)
                    visited.add(best_next)

            return tour

        def farthest_insertion(matrix: list[list[float]]) -> list[int]:
            """Farthest insertion construction."""
            n = len(matrix)
            if n < 3:
                return list(range(n))

            # Find two farthest cities
            max_dist = -1
            start_i, start_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i][j] > max_dist:
                        max_dist = matrix[i][j]
                        start_i, start_j = i, j

            tour = [start_i, start_j]
            in_tour = {start_i, start_j}

            while len(tour) < n:
                farthest_city = -1
                max_min_dist = -1

                for city in range(n):
                    if city not in in_tour:
                        min_dist = min(matrix[city][t] for t in tour)
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            farthest_city = city

                if farthest_city == -1:
                    break

                best_pos = 0
                best_increase = float("inf")

                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    increase = (
                        matrix[tour[i]][farthest_city]
                        + matrix[farthest_city][tour[j]]
                        - matrix[tour[i]][tour[j]]
                    )
                    if increase < best_increase:
                        best_increase = increase
                        best_pos = i + 1

                tour.insert(best_pos, farthest_city)
                in_tour.add(farthest_city)

            return tour

        def cheapest_insertion(matrix: list[list[float]]) -> list[int]:
            """Cheapest insertion construction."""
            n = len(matrix)
            if n < 3:
                return list(range(n))

            # Start with triangle of 3 closest cities
            min_total = float("inf")
            best_triple = (0, 1, 2)
            for i in range(min(n, 20)):
                for j in range(i + 1, min(n, 20)):
                    for k in range(j + 1, min(n, 20)):
                        total = matrix[i][j] + matrix[j][k] + matrix[k][i]
                        if total < min_total:
                            min_total = total
                            best_triple = (i, j, k)

            tour = list(best_triple)
            in_tour = set(best_triple)

            while len(tour) < n:
                best_city = -1
                best_pos = 0
                best_cost = float("inf")

                for city in range(n):
                    if city in in_tour:
                        continue
                    for i in range(len(tour)):
                        j = (i + 1) % len(tour)
                        cost = (
                            matrix[tour[i]][city]
                            + matrix[city][tour[j]]
                            - matrix[tour[i]][tour[j]]
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_city = city
                            best_pos = i + 1

                if best_city == -1:
                    break

                tour.insert(best_pos, best_city)
                in_tour.add(best_city)

            return tour

        def random_insertion(matrix: list[list[float]]) -> list[int]:
            """Random insertion construction."""
            n = len(matrix)
            if n < 3:
                return list(range(n))

            remaining = list(range(n))
            random.shuffle(remaining)

            tour = [remaining.pop()]
            if remaining:
                tour.append(remaining.pop())

            while remaining:
                city = remaining.pop()
                best_pos = 0
                best_cost = float("inf")

                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    cost = (
                        matrix[tour[i]][city]
                        + matrix[city][tour[j]]
                        - matrix[tour[i]][tour[j]]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1

                tour.insert(best_pos, city)

            return tour

        def savings_heuristic(matrix: list[list[float]]) -> list[int]:
            """Clarke-Wright savings heuristic."""
            n = len(matrix)
            if n < 2:
                return list(range(n))

            depot = 0
            savings = []

            for i in range(1, n):
                for j in range(i + 1, n):
                    s = matrix[depot][i] + matrix[depot][j] - matrix[i][j]
                    savings.append((s, i, j))

            savings.sort(reverse=True)

            routes = [[i] for i in range(1, n)]
            route_of = {i: i - 1 for i in range(1, n)}

            for s, i, j in savings:
                ri, rj = route_of[i], route_of[j]
                if ri == rj:
                    continue

                route_i = routes[ri]
                route_j = routes[rj]

                if route_i is None or route_j is None:
                    continue

                if route_i[-1] == i and route_j[0] == j:
                    new_route = route_i + route_j
                    routes[ri] = new_route
                    routes[rj] = None
                    for city in route_j:
                        route_of[city] = ri
                elif route_i[0] == i and route_j[-1] == j:
                    new_route = route_j + route_i
                    routes[rj] = new_route
                    routes[ri] = None
                    for city in route_i:
                        route_of[city] = rj
                elif route_i[-1] == i and route_j[-1] == j:
                    new_route = route_i + route_j[::-1]
                    routes[ri] = new_route
                    routes[rj] = None
                    for city in route_j:
                        route_of[city] = ri
                elif route_i[0] == i and route_j[0] == j:
                    new_route = route_i[::-1] + route_j
                    routes[ri] = new_route
                    routes[rj] = None
                    for city in route_j:
                        route_of[city] = ri

            tour = [depot]
            for route in routes:
                if route is not None:
                    tour.extend(route)

            return tour

        def christofides_construction(matrix: list[list[float]]) -> list[int]:
            """Simplified Christofides-like construction."""
            n = len(matrix)
            if n < 3:
                return list(range(n))

            # Build MST using Prim's algorithm
            in_mst = {0}
            mst_edges = []

            while len(in_mst) < n:
                best_edge = None
                best_weight = float("inf")

                for u in in_mst:
                    for v in range(n):
                        if v not in in_mst and matrix[u][v] < best_weight:
                            best_weight = matrix[u][v]
                            best_edge = (u, v)

                if best_edge:
                    mst_edges.append(best_edge)
                    in_mst.add(best_edge[1])

            # Build adjacency from MST
            adj = [[] for _ in range(n)]
            for u, v in mst_edges:
                adj[u].append(v)
                adj[v].append(u)

            # DFS to get tour
            tour = []
            visited = set()
            stack = [0]

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                tour.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            return tour

        def nearest_addition(matrix: list[list[float]]) -> list[int]:
            """Nearest addition construction."""
            n = len(matrix)
            if n < 2:
                return list(range(n))

            tour = [0]
            in_tour = {0}

            while len(tour) < n:
                nearest_city = -1
                min_dist = float("inf")

                for city in range(n):
                    if city in in_tour:
                        continue
                    for t in tour:
                        if matrix[city][t] < min_dist:
                            min_dist = matrix[city][t]
                            nearest_city = city

                if nearest_city == -1:
                    break

                # Find best position
                best_pos = 0
                best_cost = float("inf")

                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    cost = (
                        matrix[tour[i]][nearest_city]
                        + matrix[nearest_city][tour[j]]
                        - matrix[tour[i]][tour[j]]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1

                tour.insert(best_pos, nearest_city)
                in_tour.add(nearest_city)

            return tour

        def convex_hull_start(matrix: list[list[float]]) -> list[int]:
            """Start with convex hull, then insert remaining."""
            n = len(matrix)
            # Simplified: just use farthest insertion
            return farthest_insertion(matrix)

        def cluster_first(matrix: list[list[float]]) -> list[int]:
            """Cluster cities then route within clusters."""
            n = len(matrix)
            if n < 4:
                return list(range(n))

            # Simple k-means-like clustering
            k = max(2, int(math.sqrt(n / 2)))

            # Initialize centroids randomly
            centroids = random.sample(range(n), k)

            # Assign cities to nearest centroid
            clusters = [[] for _ in range(k)]
            for city in range(n):
                best_cluster = 0
                best_dist = float("inf")
                for i, c in enumerate(centroids):
                    if matrix[city][c] < best_dist:
                        best_dist = matrix[city][c]
                        best_cluster = i
                clusters[best_cluster].append(city)

            # Build tour by visiting clusters
            tour = []
            for cluster in clusters:
                if cluster:
                    # Sort within cluster by nearest neighbor
                    if tour:
                        last = tour[-1]
                        cluster.sort(key=lambda x: matrix[last][x])
                    tour.extend(cluster)

            return tour

        def sweep_algorithm(matrix: list[list[float]]) -> list[int]:
            """Sweep algorithm using angles from centroid."""
            n = len(matrix)
            if n < 2:
                return list(range(n))

            # Use city 0 as reference, sort others by "angle" (distance ratio)
            reference = 0
            others = list(range(1, n))

            # Sort by distance from reference (simplified sweep)
            others.sort(key=lambda x: matrix[reference][x])

            return [reference] + others

        def random_tour(matrix: list[list[float]]) -> list[int]:
            """Random tour construction."""
            n = len(matrix)
            tour = list(range(n))
            random.shuffle(tour)
            return tour

        # =====================================================================
        # LOCAL SEARCH PRIMITIVES (8)
        # =====================================================================

        def two_opt_improve(tour_matrix: tuple) -> list[int]:
            """2-opt improvement with adaptive limits."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Adaptive iterations
            if n <= 100:
                max_iterations = n * n
            elif n <= 500:
                max_iterations = n * int(math.sqrt(n))
            else:
                max_iterations = n * 10

            improved = True
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        if j == n - 1 and i == 0:
                            continue

                        a, b = tour[i], tour[i + 1]
                        c, d = tour[j], tour[(j + 1) % n]

                        if matrix[a][b] + matrix[c][d] > matrix[a][c] + matrix[b][d] + 1e-10:
                            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                            improved = True
                            break
                    if improved:
                        break

            return tour

        def three_opt_improve(tour_matrix: tuple) -> list[int]:
            """3-opt improvement (limited for efficiency)."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            if n < 6:
                return tour

            max_moves = min(n * 5, 500)
            moves = 0
            improved = True

            while improved and moves < max_moves:
                improved = False

                positions = random.sample(range(n - 4), min(n // 5, 50)) if n > 100 else range(n - 4)

                for i in positions:
                    if moves >= max_moves:
                        break
                    for j in range(i + 2, min(i + n // 2, n - 2)):
                        if moves >= max_moves:
                            break
                        for k in range(j + 2, min(j + n // 3, n)):
                            if k >= n:
                                continue
                            moves += 1

                            i1, j1, k1 = tour[i], tour[(i + 1) % n], tour[j]
                            j2, k2, next_k = tour[(j + 1) % n], tour[k], tour[(k + 1) % n]

                            current = matrix[i1][j1] + matrix[k1][j2] + matrix[k2][next_k]
                            new_cost = matrix[i1][k1] + matrix[j1][j2] + matrix[k2][next_k]

                            if new_cost < current - 1e-10:
                                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

            return tour

        def or_opt_improve(tour_matrix: tuple) -> list[int]:
            """Or-opt: relocate segments of 1-3 cities."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            if n < 4:
                return tour

            max_iterations = min(n * 3, 300)
            iteration = 0
            improved = True

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for seg_len in [1, 2, 3]:
                    if improved:
                        break
                    for i in range(n - seg_len):
                        if improved:
                            break

                        prev_i = (i - 1) % n
                        next_seg = (i + seg_len) % n

                        remove_cost = (
                            matrix[tour[prev_i]][tour[i]]
                            + matrix[tour[i + seg_len - 1]][tour[next_seg]]
                        )
                        reconnect_cost = matrix[tour[prev_i]][tour[next_seg]]

                        for j in range(n):
                            if i <= j <= i + seg_len:
                                continue

                            next_j = (j + 1) % n

                            insert_cost = (
                                matrix[tour[j]][tour[i]]
                                + matrix[tour[i + seg_len - 1]][tour[next_j]]
                            )
                            original_edge = matrix[tour[j]][tour[next_j]]

                            delta = (insert_cost + reconnect_cost) - (remove_cost + original_edge)

                            if delta < -1e-10:
                                segment = tour[i:i + seg_len]
                                del tour[i:i + seg_len]
                                insert_pos = j if j < i else j - seg_len + 1
                                tour[insert_pos + 1:insert_pos + 1] = segment
                                improved = True
                                break

            return tour

        def swap_improve(tour_matrix: tuple) -> list[int]:
            """Swap adjacent cities."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            improved = True
            max_iterations = min(n * 2, 200)
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(n):
                    j = (i + 1) % n
                    prev_i = (i - 1) % n
                    next_j = (j + 1) % n

                    if next_j == i:
                        continue

                    old_cost = matrix[tour[prev_i]][tour[i]] + matrix[tour[j]][tour[next_j]]
                    new_cost = matrix[tour[prev_i]][tour[j]] + matrix[tour[i]][tour[next_j]]

                    if new_cost < old_cost - 1e-10:
                        tour[i], tour[j] = tour[j], tour[i]
                        improved = True
                        break

            return tour

        def insert_improve(tour_matrix: tuple) -> list[int]:
            """Insert operator: move single city to better position."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            improved = True
            max_iterations = min(n * 2, 200)
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(n):
                    city = tour[i]
                    prev_i = (i - 1) % n
                    next_i = (i + 1) % n

                    # Cost of removing city
                    remove_cost = matrix[tour[prev_i]][city] + matrix[city][tour[next_i]]
                    bridge_cost = matrix[tour[prev_i]][tour[next_i]]

                    for j in range(n):
                        if j == i or j == prev_i:
                            continue

                        next_j = (j + 1) % n
                        if next_j == i:
                            continue

                        # Cost of inserting at position j
                        insert_cost = matrix[tour[j]][city] + matrix[city][tour[next_j]]
                        original_edge = matrix[tour[j]][tour[next_j]]

                        delta = (bridge_cost + insert_cost) - (remove_cost + original_edge)

                        if delta < -1e-10:
                            tour.pop(i)
                            new_pos = j if j < i else j - 1
                            tour.insert(new_pos + 1, city)
                            improved = True
                            break
                    if improved:
                        break

            return tour

        def invert_improve(tour_matrix: tuple) -> list[int]:
            """Invert a random segment."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Try random inversions
            for _ in range(min(n, 50)):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)

                a, b = tour[i], tour[(i + 1) % n] if i + 1 < n else tour[0]
                c, d = tour[j], tour[(j + 1) % n]

                old_cost = matrix[tour[(i - 1) % n]][tour[i]] + matrix[tour[j]][tour[(j + 1) % n]]
                new_cost = matrix[tour[(i - 1) % n]][tour[j]] + matrix[tour[i]][tour[(j + 1) % n]]

                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]

            return tour

        def lin_kernighan_improve(tour_matrix: tuple) -> list[int]:
            """Simplified Lin-Kernighan style improvement."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # LK-style: variable depth search
            max_depth = min(5, n // 10)

            for start in range(min(n, 20)):
                # Try to find improving sequence starting from each edge
                best_tour = tour[:]
                best_cost = sum(matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

                for depth in range(max_depth):
                    i = random.randint(0, n - 2)
                    j = random.randint(i + 1, n - 1)

                    new_tour = tour[:]
                    new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]

                    new_cost = sum(matrix[new_tour[k]][new_tour[(k + 1) % n]] for k in range(n))

                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost

                tour = best_tour

            return tour

        def variable_neighborhood(tour_matrix: tuple) -> list[int]:
            """Variable neighborhood descent."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix

            # Apply operators in sequence
            tour = two_opt_improve((tour, matrix))
            tour = or_opt_improve((tour, matrix))
            tour = swap_improve((tour, matrix))

            return tour

        # =====================================================================
        # PERTURBATION PRIMITIVES (6)
        # =====================================================================

        def double_bridge(tour_matrix: tuple) -> list[int]:
            """Double bridge perturbation."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            if n < 8:
                i, j = random.sample(range(n), 2)
                tour[i], tour[j] = tour[j], tour[i]
                return tour

            positions = sorted(random.sample(range(n), 4))
            p1, p2, p3, p4 = positions

            segment_a = tour[:p1 + 1]
            segment_b = tour[p1 + 1:p2 + 1]
            segment_c = tour[p2 + 1:p3 + 1]
            segment_d = tour[p3 + 1:]

            return segment_a + segment_d + segment_c + segment_b

        def segment_shuffle(tour_matrix: tuple) -> list[int]:
            """Shuffle a random segment."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            if n < 4:
                return tour

            seg_len = random.randint(max(2, n // 10), max(3, n // 4))
            start = random.randint(0, n - seg_len)

            segment = tour[start:start + seg_len]
            random.shuffle(segment)
            tour[start:start + seg_len] = segment

            return tour

        def guided_mutation(tour_matrix: tuple) -> list[int]:
            """Guided mutation based on edge costs."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Find worst edges
            edges = []
            for i in range(n):
                j = (i + 1) % n
                cost = matrix[tour[i]][tour[j]]
                edges.append((cost, i))

            edges.sort(reverse=True)

            # Perturb around worst edges
            for cost, i in edges[:3]:
                j = (i + 1) % n
                # Swap with random position
                k = random.randint(0, n - 1)
                if k != i and k != j:
                    tour[j], tour[k] = tour[k], tour[j]

            return tour

        def ruin_recreate(tour_matrix: tuple) -> list[int]:
            """Ruin and recreate perturbation."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Remove 10-20% of cities
            num_remove = max(2, n // 10)
            removed = []

            for _ in range(num_remove):
                if len(tour) > 3:
                    idx = random.randint(0, len(tour) - 1)
                    removed.append(tour.pop(idx))

            # Reinsert using cheapest insertion
            for city in removed:
                best_pos = 0
                best_cost = float("inf")

                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    cost = (
                        matrix[tour[i]][city]
                        + matrix[city][tour[j]]
                        - matrix[tour[i]][tour[j]]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1

                tour.insert(best_pos, city)

            return tour

        def large_neighborhood_search(tour_matrix: tuple) -> list[int]:
            """Large neighborhood search perturbation."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Remove 20-30% of cities
            num_remove = max(3, n // 5)

            # Remove random segment
            start = random.randint(0, n - num_remove)
            removed = tour[start:start + num_remove]
            del tour[start:start + num_remove]

            # Reinsert greedily
            for city in removed:
                best_pos = 0
                best_cost = float("inf")

                for i in range(len(tour)):
                    j = (i + 1) % len(tour)
                    cost = (
                        matrix[tour[i]][city]
                        + matrix[city][tour[j]]
                        - matrix[tour[i]][tour[j]]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = i + 1

                tour.insert(best_pos, city)

            return tour

        def adaptive_mutation(tour_matrix: tuple) -> list[int]:
            """Adaptive mutation based on tour quality."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)

            # Randomly choose mutation type
            mutation_type = random.choice(['double_bridge', 'segment_shuffle', 'guided'])

            if mutation_type == 'double_bridge':
                return double_bridge((tour, matrix))
            elif mutation_type == 'segment_shuffle':
                return segment_shuffle((tour, matrix))
            else:
                return guided_mutation((tour, matrix))

        # =====================================================================
        # META-HEURISTIC PRIMITIVES (6)
        # =====================================================================

        def simulated_annealing(tour_matrix: tuple) -> list[int]:
            """Single SA iteration with cooling."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            current_cost = sum(matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
            temp = current_cost * 0.05
            cooling = 0.95

            for _ in range(min(n * 2, 200)):
                # Generate neighbor
                i, j = sorted(random.sample(range(n), 2))
                new_tour = tour[:]
                new_tour[i:j + 1] = new_tour[i:j + 1][::-1]

                new_cost = sum(matrix[new_tour[k]][new_tour[(k + 1) % n]] for k in range(n))
                delta = new_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                    tour = new_tour
                    current_cost = new_cost

                temp *= cooling

            return tour

        def tabu_search(tour_matrix: tuple) -> list[int]:
            """Simple tabu search."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            tabu_list = set()
            tabu_tenure = max(5, int(math.sqrt(n)))

            current_cost = sum(matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
            best_tour = tour[:]
            best_cost = current_cost

            for iteration in range(min(n * 2, 200)):
                best_neighbor = None
                best_neighbor_cost = float("inf")

                # Try all 2-opt neighbors
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        if (i, j) in tabu_list:
                            continue

                        new_tour = tour[:]
                        new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
                        new_cost = sum(matrix[new_tour[k]][new_tour[(k + 1) % n]] for k in range(n))

                        if new_cost < best_neighbor_cost:
                            best_neighbor = new_tour
                            best_neighbor_cost = new_cost
                            best_move = (i, j)

                        if len(list(range(i, j))) > 10:
                            break
                    if best_neighbor and best_neighbor_cost < current_cost:
                        break

                if best_neighbor:
                    tour = best_neighbor
                    current_cost = best_neighbor_cost
                    tabu_list.add(best_move)

                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop()

                    if current_cost < best_cost:
                        best_tour = tour[:]
                        best_cost = current_cost

            return best_tour

        def genetic_crossover(tour_matrix: tuple) -> list[int]:
            """Order crossover with random parent."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)

            # Create random second parent
            parent2 = list(range(n))
            random.shuffle(parent2)

            # Order crossover
            start = random.randint(0, n - 2)
            end = random.randint(start + 1, n - 1)

            child = [-1] * n
            child[start:end + 1] = tour[start:end + 1]

            remaining = [x for x in parent2 if x not in child]

            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = remaining[idx]
                    idx += 1

            return child

        def ant_colony(tour_matrix: tuple) -> list[int]:
            """Single ant construction with pheromone bias."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)

            # Use current tour to bias construction
            # (simplified: nearest neighbor with randomization)
            start = random.randint(0, n - 1)
            new_tour = [start]
            visited = {start}

            while len(new_tour) < n:
                current = new_tour[-1]

                # Calculate probabilities
                unvisited = [c for c in range(n) if c not in visited]
                if not unvisited:
                    break

                # Bias towards short edges and edges in current tour
                weights = []
                for city in unvisited:
                    dist = matrix[current][city]
                    # Check if edge is in original tour
                    bonus = 1.0
                    for i in range(n):
                        if tour[i] == current and tour[(i + 1) % n] == city:
                            bonus = 2.0
                            break
                    weights.append(bonus / (dist + 0.01))

                total = sum(weights)
                probs = [w / total for w in weights]

                # Roulette wheel selection
                r = random.random()
                cumsum = 0
                chosen = unvisited[0]
                for city, prob in zip(unvisited, probs):
                    cumsum += prob
                    if r <= cumsum:
                        chosen = city
                        break

                new_tour.append(chosen)
                visited.add(chosen)

            return new_tour

        def particle_swarm(tour_matrix: tuple) -> list[int]:
            """PSO-inspired tour update."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix
            n = len(tour)
            tour = list(tour)

            # Generate "velocity" as sequence of swaps
            num_swaps = random.randint(1, max(2, n // 10))

            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                tour[i], tour[j] = tour[j], tour[i]

            return tour

        def iterated_local_search(tour_matrix: tuple) -> list[int]:
            """ILS: perturb then local search."""
            if isinstance(tour_matrix, list):
                return tour_matrix

            tour, matrix = tour_matrix

            # Perturb
            tour = double_bridge((tour, matrix))

            # Local search
            tour = two_opt_improve((tour, matrix))

            return tour

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
        pset.addPrimitive(lambda x: (random_tour(x), x), 1, name="random_tour")

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

    def _validate_tour(self, tour: Any, n: int) -> bool:
        """Validate tour."""
        if isinstance(tour, tuple):
            tour = tour[0]

        if not isinstance(tour, list):
            return False
        if len(tour) != n:
            return False
        if set(tour) != set(range(n)):
            return False
        return True

    def _calculate_tour_cost(self, tour: Any, matrix: list[list[float]]) -> float:
        """Calculate tour cost."""
        if isinstance(tour, tuple):
            tour = tour[0]

        n = len(tour)
        return sum(matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
