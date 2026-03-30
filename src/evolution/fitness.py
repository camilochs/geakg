"""Fitness evaluation for algorithms.

Provides fitness functions that execute algorithm operator sequences
on problem instances.
"""

import random
from typing import Any

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorCategory, OperatorNode
from src.domains.tsp import TSPDomain, TSPInstance, TSPSolution
from src.evolution.population import Algorithm


class TSPFitnessEvaluator:
    """Evaluates algorithm fitness on TSP instances.

    Executes the operator sequence defined by the algorithm
    and returns the resulting tour cost.
    """

    def __init__(self, akg: AlgorithmicKnowledgeGraph) -> None:
        """Initialize evaluator.

        Args:
            akg: Algorithmic Knowledge Graph for operator info
        """
        self.akg = akg
        self.domain = TSPDomain()

    def evaluate(self, algorithm: Algorithm, instance: TSPInstance) -> float:
        """Evaluate algorithm on TSP instance.

        Args:
            algorithm: Algorithm to evaluate
            instance: TSP instance

        Returns:
            Tour cost (lower is better)
        """
        if not algorithm.operators:
            return float("inf")

        # Execute operator sequence
        solution = self._execute_operators(algorithm.operators, instance)

        if solution is None:
            return float("inf")

        return solution.cost

    def _execute_operators(
        self, operators: list[str], instance: TSPInstance
    ) -> TSPSolution | None:
        """Execute operator sequence on instance.

        Args:
            operators: List of operator IDs
            instance: TSP instance

        Returns:
            Final solution or None if failed
        """
        solution = None

        for op_id in operators:
            node = self.akg.get_node(op_id)
            # Check for OperatorNode by attribute (avoids import path issues)
            if not hasattr(node, 'category'):
                continue

            try:
                if node.category == OperatorCategory.CONSTRUCTION:
                    solution = self._apply_construction(op_id, instance)
                elif node.category == OperatorCategory.LOCAL_SEARCH and solution:
                    solution = self._apply_local_search(op_id, solution, instance)
                elif node.category == OperatorCategory.PERTURBATION and solution:
                    solution = self._apply_perturbation(op_id, solution, instance)
                elif node.category == OperatorCategory.META_HEURISTIC and solution:
                    solution = self._apply_meta(op_id, solution, instance)
            except Exception:
                # If operator fails, continue with current solution
                pass

        return solution

    def _apply_construction(
        self, op_id: str, instance: TSPInstance
    ) -> TSPSolution:
        """Apply construction operator.

        Args:
            op_id: Operator ID
            instance: TSP instance

        Returns:
            Initial solution
        """
        if op_id == "greedy_nearest_neighbor":
            return self.domain.nearest_neighbor_solution(instance)
        elif op_id == "farthest_insertion":
            return self.domain.farthest_insertion_solution(instance)
        elif op_id == "cheapest_insertion":
            return self.domain.cheapest_insertion_solution(instance)
        elif op_id == "nearest_addition":
            return self.domain.cheapest_insertion_solution(instance)  # Similar approach
        elif op_id == "random_insertion":
            return self.domain.random_solution(instance)
        elif op_id == "savings_heuristic":
            return self.domain.savings_solution(instance)
        elif op_id == "convex_hull_start":
            return self.domain.convex_hull_solution(instance)
        elif op_id == "christofides_construction":
            # Christofides needs MST - approximate with farthest insertion
            return self.domain.farthest_insertion_solution(instance)
        elif op_id == "cluster_first":
            # Cluster-first: use savings as approximation
            return self.domain.savings_solution(instance)
        elif op_id == "sweep_algorithm":
            # Sweep: use convex hull as approximation
            return self.domain.convex_hull_solution(instance)
        else:
            return self.domain.random_solution(instance)

    def _apply_local_search(
        self, op_id: str, solution: TSPSolution, instance: TSPInstance
    ) -> TSPSolution:
        """Apply local search operator.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: TSP instance

        Returns:
            Improved solution
        """
        if op_id == "two_opt":
            return self.domain.two_opt_improve(solution, instance, max_iterations=500)
        elif op_id == "three_opt":
            return self.domain.three_opt_improve(solution, instance, max_iterations=50)
        elif op_id == "or_opt":
            return self.domain.or_opt_improve(solution, instance, max_iterations=200)
        elif op_id == "swap":
            # Swap two cities
            tour = solution.tour.copy()
            n = len(tour)
            best_tour = tour
            best_cost = solution.cost
            for _ in range(100):
                i, j = random.sample(range(n), 2)
                tour[i], tour[j] = tour[j], tour[i]
                cost = self.domain.evaluate_solution(TSPSolution(tour=tour), instance)
                if cost < best_cost:
                    best_tour = tour.copy()
                    best_cost = cost
                else:
                    tour[i], tour[j] = tour[j], tour[i]  # Revert
            result = TSPSolution(tour=best_tour)
            result.cost = best_cost
            return result
        elif op_id == "insert":
            # Remove and insert at best position
            tour = solution.tour.copy()
            n = len(tour)
            for _ in range(50):
                i = random.randint(0, n - 1)
                city = tour.pop(i)
                best_pos, best_cost = i, float("inf")
                for pos in range(n):
                    tour.insert(pos, city)
                    cost = self.domain.evaluate_solution(TSPSolution(tour=tour), instance)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                    tour.pop(pos)
                tour.insert(best_pos, city)
            result = TSPSolution(tour=tour)
            result.cost = self.domain.evaluate_solution(result, instance)
            return result
        elif op_id == "invert":
            # Invert random segment
            tour = solution.tour.copy()
            n = len(tour)
            best_tour = tour
            best_cost = solution.cost
            for _ in range(100):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                tour[i:j+1] = tour[i:j+1][::-1]
                cost = self.domain.evaluate_solution(TSPSolution(tour=tour), instance)
                if cost < best_cost:
                    best_tour = tour.copy()
                    best_cost = cost
                else:
                    tour[i:j+1] = tour[i:j+1][::-1]  # Revert
            result = TSPSolution(tour=best_tour)
            result.cost = best_cost
            return result
        elif op_id == "lin_kernighan":
            return self.domain.lin_kernighan_improve(solution, instance, max_iterations=30)
        elif op_id == "variable_neighborhood":
            # VND: cycle through neighborhoods
            improved = solution
            improved = self.domain.two_opt_improve(improved, instance, max_iterations=200)
            improved = self.domain.or_opt_improve(improved, instance, max_iterations=100)
            improved = self.domain.three_opt_improve(improved, instance, max_iterations=30)
            return improved
        else:
            return solution

    def _apply_perturbation(
        self, op_id: str, solution: TSPSolution, instance: TSPInstance
    ) -> TSPSolution:
        """Apply perturbation operator.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: TSP instance

        Returns:
            Perturbed solution
        """
        tour = solution.tour.copy()
        n = len(tour)

        if op_id == "double_bridge":
            # Double bridge move
            if n >= 8:
                # Select 4 positions
                positions = sorted(random.sample(range(n), 4))
                p1, p2, p3, p4 = positions

                # Reconnect segments
                new_tour = (
                    tour[:p1+1] +
                    tour[p3+1:p4+1] +
                    tour[p2+1:p3+1] +
                    tour[p1+1:p2+1] +
                    tour[p4+1:]
                )
                tour = new_tour

        elif op_id == "random_segment_shuffle":
            # Shuffle random segments
            if n >= 4:
                seg_size = n // 4
                segments = [tour[i:i+seg_size] for i in range(0, n, seg_size)]
                random.shuffle(segments)
                tour = [city for seg in segments for city in seg]

        elif op_id in ("ruin_recreate", "large_neighborhood_search"):
            # Remove 30% of cities and reinsert
            ruin_size = max(2, n // 3)
            removed_indices = random.sample(range(n), ruin_size)
            removed = [tour[i] for i in removed_indices]
            kept = [tour[i] for i in range(n) if i not in removed_indices]

            # Reinsert at best positions
            for city in removed:
                best_pos = 0
                best_cost = float("inf")
                for pos in range(len(kept) + 1):
                    test_tour = kept[:pos] + [city] + kept[pos:]
                    cost = self._calculate_tour_cost(test_tour, instance)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                kept.insert(best_pos, city)
            tour = kept

        elif op_id in ("guided_mutation", "adaptive_mutation"):
            # Random segment reversal
            if n >= 4:
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                tour[i:j+1] = tour[i:j+1][::-1]

        result = TSPSolution(tour=tour)
        result.cost = self.domain.evaluate_solution(result, instance)
        return result

    def _apply_meta(
        self, op_id: str, solution: TSPSolution, instance: TSPInstance
    ) -> TSPSolution:
        """Apply meta-heuristic step.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: TSP instance

        Returns:
            Modified solution
        """
        if op_id == "simulated_annealing_step":
            # Single SA step with 2-opt move
            return self.domain.two_opt_improve(solution, instance, max_iterations=100)

        elif op_id == "tabu_search_step":
            # Single tabu step (use 2-opt)
            return self.domain.two_opt_improve(solution, instance, max_iterations=100)

        elif op_id == "iterated_local_search":
            # ILS step: perturb + local search
            perturbed = self._apply_perturbation("double_bridge", solution, instance)
            return self.domain.two_opt_improve(perturbed, instance, max_iterations=500)

        elif op_id == "genetic_crossover":
            # No crossover without population - just improve
            return self.domain.two_opt_improve(solution, instance, max_iterations=100)

        else:
            return solution

    def _calculate_tour_cost(
        self, tour: list[int], instance: TSPInstance
    ) -> float:
        """Calculate tour cost.

        Args:
            tour: Tour as list of city indices
            instance: TSP instance

        Returns:
            Total tour cost
        """
        total = 0.0
        n = len(tour)
        for i in range(n):
            from_city = tour[i]
            to_city = tour[(i + 1) % n]
            total += instance.distance_matrix[from_city][to_city]
        return total


def create_tsp_fitness_function(
    akg: AlgorithmicKnowledgeGraph,
) -> callable:
    """Create fitness function for TSP.

    Args:
        akg: Algorithmic Knowledge Graph

    Returns:
        Fitness function(algorithm, instance) -> float
    """
    evaluator = TSPFitnessEvaluator(akg)
    return evaluator.evaluate


class JSSPFitnessEvaluator:
    """Evaluates algorithm fitness on JSSP instances.

    Executes the operator sequence defined by the algorithm
    and returns the resulting makespan.
    """

    def __init__(self, akg: AlgorithmicKnowledgeGraph) -> None:
        """Initialize evaluator.

        Args:
            akg: Algorithmic Knowledge Graph for operator info
        """
        from src.domains.jssp import JSSPDomain

        self.akg = akg
        self.domain = JSSPDomain()

    def evaluate(self, algorithm: Algorithm, instance: Any) -> float:
        """Evaluate algorithm on JSSP instance.

        Args:
            algorithm: Algorithm to evaluate
            instance: JSSP instance

        Returns:
            Makespan (lower is better)
        """
        # Check for JSSP instance by attributes (avoids import path issues)
        if not hasattr(instance, 'n_jobs') or not hasattr(instance, 'n_machines'):
            return float("inf")

        if not algorithm.operators:
            return float("inf")

        # Execute operator sequence
        solution = self._execute_operators(algorithm.operators, instance)

        if solution is None:
            return float("inf")

        return float(solution.makespan)

    def _execute_operators(
        self, operators: list[str], instance: Any
    ) -> Any:
        """Execute operator sequence on JSSP instance.

        Args:
            operators: List of operator IDs
            instance: JSSP instance

        Returns:
            Final solution or None if failed
        """
        from src.domains.jssp import JSSPSolution

        solution = None

        for op_id in operators:
            node = self.akg.get_node(op_id)
            # Check for OperatorNode by attribute (avoids import path issues)
            if not hasattr(node, 'category'):
                continue

            try:
                if node.category == OperatorCategory.CONSTRUCTION:
                    solution = self._apply_construction(op_id, instance)
                elif node.category == OperatorCategory.LOCAL_SEARCH and solution:
                    solution = self._apply_local_search(op_id, solution, instance)
                elif node.category == OperatorCategory.PERTURBATION and solution:
                    solution = self._apply_perturbation(op_id, solution, instance)
                elif node.category == OperatorCategory.META_HEURISTIC and solution:
                    solution = self._apply_meta(op_id, solution, instance)
            except Exception:
                pass

        return solution

    def _apply_construction(self, op_id: str, instance: Any) -> Any:
        """Apply construction operator for JSSP.

        Maps TSP construction heuristics to JSSP equivalents.

        Args:
            op_id: Operator ID
            instance: JSSP instance

        Returns:
            Initial schedule
        """
        # Map TSP construction operators to JSSP priority dispatch rules
        if op_id in ("greedy_nearest_neighbor", "cheapest_insertion", "savings_heuristic"):
            # Greedy-like: use shortest processing time
            return self.domain.priority_dispatch_solution(instance, priority="spt")
        elif op_id in ("farthest_insertion", "christofides_construction"):
            # Anti-greedy: use longest processing time
            return self.domain.priority_dispatch_solution(instance, priority="lpt")
        elif op_id in ("random_insertion", "cluster_first", "sweep_algorithm"):
            return self.domain.random_solution(instance)
        else:
            # Default: random dispatch
            return self.domain.random_solution(instance)

    def _apply_local_search(
        self, op_id: str, solution: Any, instance: Any
    ) -> Any:
        """Apply local search for JSSP.

        Adapts TSP local search concepts to JSSP.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: JSSP instance

        Returns:
            Improved solution
        """
        from src.domains.jssp import JSSPSolution

        schedule = solution.schedule.copy()
        n_ops = len(schedule)
        improved = True

        if op_id in ("two_opt", "three_opt", "lin_kernighan"):
            # Swap adjacent operations on critical path (simplified)
            max_iter = 100 if op_id == "two_opt" else 200
            for _ in range(max_iter):
                if not improved:
                    break
                improved = False
                for i in range(n_ops - 1):
                    # Try swapping adjacent ops if they're from different jobs
                    job1, _ = schedule[i]
                    job2, _ = schedule[i + 1]
                    if job1 != job2:
                        # Swap
                        schedule[i], schedule[i + 1] = schedule[i + 1], schedule[i]
                        # Check if valid and better
                        if self._is_valid_schedule(schedule, instance):
                            new_makespan = self.domain._compute_makespan(schedule, instance)
                            if new_makespan < solution.makespan:
                                improved = True
                                solution = JSSPSolution(schedule=schedule)
                                solution.makespan = new_makespan
                            else:
                                # Revert
                                schedule[i], schedule[i + 1] = schedule[i + 1], schedule[i]
                        else:
                            # Revert invalid swap
                            schedule[i], schedule[i + 1] = schedule[i + 1], schedule[i]

        elif op_id in ("or_opt", "swap", "insert"):
            # Block moves
            for _ in range(50):
                if n_ops < 4:
                    break
                i = random.randint(0, n_ops - 2)
                j = random.randint(0, n_ops - 1)
                if i != j and abs(i - j) > 1:
                    # Try moving operation i to position j
                    op = schedule.pop(i)
                    schedule.insert(j, op)
                    if self._is_valid_schedule(schedule, instance):
                        new_makespan = self.domain._compute_makespan(schedule, instance)
                        if new_makespan < solution.makespan:
                            solution = JSSPSolution(schedule=schedule)
                            solution.makespan = new_makespan
                        else:
                            # Revert
                            schedule.pop(j if j < i else j - 1)
                            schedule.insert(i, op)
                    else:
                        # Revert
                        schedule.pop(j if j < i else j - 1)
                        schedule.insert(i, op)

        return solution

    def _apply_perturbation(
        self, op_id: str, solution: Any, instance: Any
    ) -> Any:
        """Apply perturbation for JSSP.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: JSSP instance

        Returns:
            Perturbed solution
        """
        from src.domains.jssp import JSSPSolution

        schedule = solution.schedule.copy()
        n_ops = len(schedule)

        if op_id in ("double_bridge", "random_segment_shuffle"):
            # Shuffle segments of the schedule
            if n_ops >= 8:
                seg_size = n_ops // 4
                segments = []
                for i in range(0, n_ops, seg_size):
                    segments.append(schedule[i:i + seg_size])
                random.shuffle(segments)
                schedule = [op for seg in segments for op in seg]

        elif op_id in ("ruin_recreate", "large_neighborhood_search"):
            # Remove random operations and rebuild
            n_remove = max(2, n_ops // 4)
            # Remove some operations (track by job)
            removed = []
            for _ in range(n_remove):
                if len(schedule) > instance.n_jobs:
                    idx = random.randint(0, len(schedule) - 1)
                    removed.append(schedule.pop(idx))

            # Re-insert at random valid positions
            for op in removed:
                valid_positions = self._find_valid_positions(op, schedule, instance)
                if valid_positions:
                    pos = random.choice(valid_positions)
                    schedule.insert(pos, op)
                else:
                    schedule.append(op)

        elif op_id in ("guided_mutation", "adaptive_mutation"):
            # Random adjacent swaps
            n_swaps = max(1, n_ops // 10)
            for _ in range(n_swaps):
                if n_ops >= 2:
                    i = random.randint(0, n_ops - 2)
                    job1, _ = schedule[i]
                    job2, _ = schedule[i + 1]
                    if job1 != job2:
                        schedule[i], schedule[i + 1] = schedule[i + 1], schedule[i]

        # Validate and repair if needed
        if not self._is_valid_schedule(schedule, instance):
            # Repair by regenerating
            return self.domain.random_solution(instance)

        result = JSSPSolution(schedule=schedule)
        result.makespan = self.domain._compute_makespan(schedule, instance)
        return result

    def _apply_meta(
        self, op_id: str, solution: Any, instance: Any
    ) -> Any:
        """Apply meta-heuristic step for JSSP.

        Args:
            op_id: Operator ID
            solution: Current solution
            instance: JSSP instance

        Returns:
            Modified solution
        """
        if op_id == "simulated_annealing_step":
            # SA: try perturbation, accept if better or with probability
            perturbed = self._apply_perturbation("guided_mutation", solution, instance)
            if perturbed.makespan <= solution.makespan:
                return perturbed
            # Accept worse with small probability
            if random.random() < 0.1:
                return perturbed
            return solution

        elif op_id == "tabu_search_step":
            # Tabu: local search with diversification
            improved = self._apply_local_search("swap", solution, instance)
            return improved

        elif op_id == "iterated_local_search":
            # ILS: perturb + local search
            perturbed = self._apply_perturbation("double_bridge", solution, instance)
            return self._apply_local_search("two_opt", perturbed, instance)

        elif op_id == "genetic_crossover":
            # No crossover without population
            return self._apply_local_search("swap", solution, instance)

        return solution

    def _is_valid_schedule(
        self, schedule: list[tuple[int, int]], instance: Any
    ) -> bool:
        """Check if schedule respects precedence constraints.

        Args:
            schedule: List of (job_id, op_idx) tuples
            instance: JSSP instance

        Returns:
            True if valid
        """
        # Track next expected operation for each job
        next_op = [0] * instance.n_jobs

        for job_id, op_idx in schedule:
            if op_idx != next_op[job_id]:
                return False
            next_op[job_id] += 1

        return True

    def _find_valid_positions(
        self,
        op: tuple[int, int],
        schedule: list[tuple[int, int]],
        instance: Any,
    ) -> list[int]:
        """Find valid insertion positions for an operation.

        Args:
            op: Operation (job_id, op_idx) to insert
            schedule: Current schedule
            instance: JSSP instance

        Returns:
            List of valid positions
        """
        job_id, op_idx = op
        valid = []

        for pos in range(len(schedule) + 1):
            # Check if inserting here maintains precedence
            test_schedule = schedule[:pos] + [op] + schedule[pos:]
            if self._is_valid_schedule(test_schedule, instance):
                valid.append(pos)

        return valid


def create_jssp_fitness_function(
    akg: AlgorithmicKnowledgeGraph,
) -> callable:
    """Create fitness function for JSSP.

    Args:
        akg: Algorithmic Knowledge Graph

    Returns:
        Fitness function(algorithm, instance) -> float
    """
    evaluator = JSSPFitnessEvaluator(akg)
    return evaluator.evaluate
