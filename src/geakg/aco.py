"""ACO-based Traversal for Algorithmic Knowledge Graphs.

Implements Ant Colony Optimization for selecting operator sequences from the AKG.
Instead of greedy selection, ants probabilistically explore the operator graph,
guided by pheromones (historical success) and heuristics (operator quality).

Key concepts:
- Nodes = Operators (not cities)
- Pheromones = "This operator sequence produces good algorithms"
- Heuristic = LLM-assigned initial weights (semantic bias)
- Energy Budget = Maximum steps to prevent infinite loops

Level 3 Extension: Conditional Transitions
- Edges can have conditions (e.g., "after 3 generations without improvement")
- Conditions are evaluated against ExecutionContext
- When conditions are met, edge probability is boosted by condition_boost
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorCategory

if TYPE_CHECKING:
    from src.geakg.conditions import ExecutionContext
    from src.geakg.core.role_schema import RoleSchema


# =============================================================================
# INCOMPATIBILITY TRACKER (Symbolic Reasoning)
# =============================================================================

@dataclass
class IncompatibilityTracker:
    """Symbolic reasoning: track bad operator transitions.

    This tracker detects operator sequences that frequently fail and penalizes
    those transitions in future selections. It implements pure symbolic reasoning
    (no LLM) based on:

    - **Input**: Failed/successful paths (structured data)
    - **Process**: Frequency counting of successes and failures per transition
    - **Rule**: If transition (A→B) fails >X% of the time → mark as incompatible
    - **Action**: Reduce probability of that transition (deterministic penalty)
    """

    failure_threshold: float = 0.5  # >50% failure rate = incompatible
    min_samples: int = 10  # Minimum uses of a transition before penalizing
    penalty_factor: float = 0.3  # Multiply probability by this (0.3 = 70% reduction)

    # Internal counters (not configurable)
    _transition_failures: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _transition_successes: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _total_failures: int = field(default=0)
    _total_successes: int = field(default=0)
    _logged_incompatible: set[tuple[str, str]] = field(default_factory=set)

    def record_path(self, path: list[str], is_failure: bool) -> None:
        """Record path outcome for transition tracking.

        Args:
            path: Sequence of operator/role IDs.
            is_failure: True if the path resulted in a poor solution.
        """
        if len(path) < 2:
            return

        for i in range(len(path) - 1):
            transition = (path[i], path[i + 1])
            if is_failure:
                self._transition_failures[transition] += 1
            else:
                self._transition_successes[transition] += 1

        if is_failure:
            self._total_failures += 1
        else:
            self._total_successes += 1

    def get_penalty(self, source: str, target: str) -> float:
        """Get penalty multiplier for a transition.

        Returns 1.0 (no penalty) if:
        - Not enough data (< min_samples uses of this transition)
        - Transition failure rate below threshold

        Returns penalty_factor (e.g., 0.3) if transition is incompatible.

        Args:
            source: Source operator/role ID.
            target: Target operator/role ID.

        Returns:
            Penalty multiplier (0.3-1.0).
        """
        transition = (source, target)
        fail_count = self._transition_failures.get(transition, 0)
        success_count = self._transition_successes.get(transition, 0)
        total_uses = fail_count + success_count

        if total_uses < self.min_samples:
            return 1.0  # Not enough data for this transition

        if fail_count == 0:
            return 1.0  # Never failed

        # Calculate failure rate: what % of this transition's uses failed?
        fail_rate = fail_count / total_uses

        if fail_rate > self.failure_threshold:
            # Log only once per transition
            if transition not in self._logged_incompatible:
                self._logged_incompatible.add(transition)
                logger.info(
                    f"[INCOMPATIBLE] {source} → {target} "
                    f"(fail rate {fail_rate:.0%}, {fail_count}/{total_uses} uses)"
                )
            return self.penalty_factor

        return 1.0

    def get_incompatible_transitions(self) -> set[tuple[str, str]]:
        """Get all transitions currently marked as incompatible.

        Returns:
            Set of (source, target) tuples that are penalized.
        """
        incompatible = set()

        # Check all transitions that have been seen
        all_transitions = set(self._transition_failures.keys()) | set(self._transition_successes.keys())

        for transition in all_transitions:
            fail_count = self._transition_failures.get(transition, 0)
            success_count = self._transition_successes.get(transition, 0)
            total_uses = fail_count + success_count

            if total_uses < self.min_samples:
                continue  # Not enough data

            fail_rate = fail_count / total_uses
            if fail_rate > self.failure_threshold:
                incompatible.add(transition)

        return incompatible

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about tracked transitions.

        Returns:
            Dictionary with tracking statistics.
        """
        return {
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "unique_transitions": len(self._transition_failures),
            "incompatible_count": len(self.get_incompatible_transitions()),
        }


# =============================================================================
class OperatorMode(str, Enum):
    """Operating mode for operator selection within roles.

    STATIC: Current behavior - uniform random selection, no pheromone learning
            at operator level. Backward compatible.
    DYNAMIC: ACO pheromones at operator level + optional LLM synthesis
             of new operators when MMAS signals bottlenecks.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class ACOConfig:
    """Configuration for ACO traversal."""

    # ACO parameters
    alpha: float = 2.0  # Pheromone importance (higher = more greedy towards best)
    beta: float = 2.0   # Heuristic importance (LLM weight)
    rho: float = 0.1    # Evaporation rate
    q: float = 100.0    # Pheromone deposit constant

    # Colony parameters
    n_ants: int = 10    # Number of ants per iteration

    # Energy budget (prevents infinite loops)
    max_steps: int = 8  # Maximum operators in sequence
    energy_costs: dict[str, float] = field(default_factory=lambda: {
        "construction": 1.0,
        "local_search": 1.0,
        "perturbation": 1.0,
    })
    initial_energy: float = 9.0  # Starting energy for each ant (~6 ops)

    # Variable energy per ant (enables exploration of different path lengths)
    variable_energy: bool = True  # Each ant gets random energy in range
    energy_min: float = 4.0  # Minimum energy (~3 operators)
    energy_max: float = 12.0  # Maximum energy (~8 operators)

    # Exploration
    exploration_rate: float = 0.1  # Probability of random choice
    min_pheromone: float = 0.001  # Lower min for more differentiation
    max_pheromone: float = 1.0  # MMAS normalized range [0.001, 1.0]

    # MMAS (Min-Max Ant System) parameters
    use_mmas: bool = True  # Use MMAS variant
    mmas_best_only: bool = True  # Only best ant deposits pheromone
    mmas_global_best: bool = False  # Use global best (True) or iteration best (False)
    stagnation_limit: int = 20  # Iterations without improvement before pheromone reset

    # Variable length paths (energy is the primary control)
    allow_early_stop: bool = False  # Disabled - energy controls path length
    stop_probability: float = 0.05  # Low probability if re-enabled
    min_steps: int = 3  # Minimum steps before allowing stop
    length_bonus: float = 0.1  # Bonus multiplier for shorter paths (0 = no bonus)

    # Level 3: Conditional transitions
    enable_conditions: bool = True  # Set False to disable condition evaluation (ablation)

    # Operator pruning (remove underperforming operators to speed up selection)
    enable_pruning: bool = True  # Enable automatic pruning of bad operators
    pruning_min_uses: int = 15  # Minimum uses before considering for pruning
    pruning_improvement_threshold: float = 0.03  # Prune if improvement rate < 3%
    pruning_pheromone_threshold: float = 0.08  # Prune if pheromone < 8% of max
    pruning_grace_period: int = 75  # Synthesized operators protected for N iterations
    pruning_check_interval: int = 30  # Check for pruning every N iterations


@dataclass
class Ant:
    """An ant that traverses the AKG."""

    path: list[str] = field(default_factory=list)
    energy: float = 10.0
    fitness: float | None = None
    gap: float | None = None  # Gap percentage from optimal (if known)

    def can_continue(self, cost: float, max_steps: int = 0) -> bool:
        """Check if ant can continue traversal.

        Path length is determined purely by energy budget.
        max_steps parameter kept for API compatibility but ignored.
        """
        return self.energy >= cost

    def move_to(self, operator_id: str, cost: float) -> None:
        """Move to next operator, consuming energy."""
        self.path.append(operator_id)
        self.energy -= cost


class ACOSelector:
    """ACO-based operator sequence selector.

    Uses ant colony optimization to select operator sequences from the AKG.
    Combines pheromone trails (learned from good solutions) with heuristic
    information (LLM-assigned edge weights).

    Level 3 (Conditional Transitions):
    When enable_conditions=True and execution_context is set, edges with
    conditions are evaluated. If conditions are met, the edge probability
    is multiplied by condition_boost (BOOST mode, not GATE mode).
    """

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        config: ACOConfig | None = None,
    ) -> None:
        """Initialize ACO selector.

        Args:
            akg: The Algorithmic Knowledge Graph
            config: ACO configuration
        """
        self.akg = akg
        self.config = config or ACOConfig()

        # Dynamic max_steps based on AKG size
        # Typical metaheuristics use 3-8 operators, scale with graph size
        # Formula: min(12, max(4, n_nodes // 4))
        n_nodes = len(akg.nodes)
        dynamic_max_steps = min(12, max(4, n_nodes // 4))

        # Only override if using default value (8)
        if self.config.max_steps == 8:
            # Create new config with dynamic max_steps
            self.config = ACOConfig(
                alpha=self.config.alpha,
                beta=self.config.beta,
                rho=self.config.rho,
                q=self.config.q,
                n_ants=self.config.n_ants,
                max_steps=dynamic_max_steps,
                energy_costs=self.config.energy_costs,
                initial_energy=self.config.initial_energy,
                variable_energy=self.config.variable_energy,
                energy_min=self.config.energy_min,
                energy_max=self.config.energy_max,
                exploration_rate=self.config.exploration_rate,
                min_pheromone=self.config.min_pheromone,
                max_pheromone=self.config.max_pheromone,
                use_mmas=self.config.use_mmas,
                mmas_best_only=self.config.mmas_best_only,
                mmas_global_best=self.config.mmas_global_best,
                stagnation_limit=self.config.stagnation_limit,
                allow_early_stop=self.config.allow_early_stop,
                stop_probability=self.config.stop_probability,
                min_steps=self.config.min_steps,
                length_bonus=self.config.length_bonus,
                enable_conditions=self.config.enable_conditions,
            )
            logger.debug(f"Dynamic max_steps: {dynamic_max_steps} (AKG has {n_nodes} nodes)")

        # Initialize pheromone matrix (separate from edge weights)
        # Edge weights = heuristic (η), Pheromones = learned (τ)
        self.pheromones: dict[tuple[str, str], float] = {}
        self._initialize_pheromones()

        # Statistics
        self.iterations = 0
        self.best_path: list[str] = []
        self.best_fitness: float = float("inf")

        # MMAS tracking
        self.stagnation_counter = 0
        self.last_best_fitness: float = float("inf")

        # Level 3: Execution context for condition evaluation
        self._execution_context: ExecutionContext | None = None

    def set_execution_context(self, context: ExecutionContext | None) -> None:
        """Set the execution context for condition evaluation.

        The execution context provides runtime metrics (stagnation, gap, diversity)
        used to evaluate edge conditions.

        Args:
            context: Current execution state, or None to disable conditions
        """
        self._execution_context = context

    def get_execution_context(self) -> ExecutionContext | None:
        """Get the current execution context."""
        return self._execution_context

    def _initialize_pheromones(self) -> None:
        """Initialize pheromone levels from LLM-assigned edge weights."""
        for (source, target), edge in self.akg.edges.items():
            # Start with LLM weight as initial pheromone
            # This gives semantic bias from the beginning
            initial = max(self.config.min_pheromone, edge.weight)
            self.pheromones[(source, target)] = initial

    def _get_heuristic(self, source: str, target: str) -> float:
        """Get heuristic value (η) for edge - based on LLM weight."""
        edge = self.akg.edges.get((source, target))
        if edge:
            return max(0.01, edge.weight)
        return 0.01

    def _get_pheromone(self, source: str, target: str) -> float:
        """Get pheromone level (τ) for edge."""
        return self.pheromones.get((source, target), self.config.min_pheromone)

    def _get_energy_cost(self, operator_id: str) -> float:
        """Get energy cost for using an operator."""
        node = self.akg.get_node(operator_id)
        if node and hasattr(node, 'category'):
            category = node.category.value
            return self.config.energy_costs.get(category, 1.0)
        return 1.0

    def _get_valid_next_operators(self, current: str | None) -> list[str]:
        """Get valid next operators from current position."""
        if current is None:
            # Start with construction operators
            return [
                n.id for n in self.akg.get_operators_by_category(
                    OperatorCategory.CONSTRUCTION
                )
            ]
        return self.akg.get_valid_transitions(current)

    def _get_condition_boost(self, source: str, target: str) -> float:
        """Get condition boost for an edge.

        If conditions are enabled and the edge has conditions, evaluate them.
        Returns the condition_boost if conditions are met, 1.0 otherwise.

        Args:
            source: Source operator ID
            target: Target operator ID

        Returns:
            Boost multiplier (1.0 if no conditions or conditions not met)
        """
        # Skip if conditions are disabled
        if not self.config.enable_conditions:
            return 1.0

        # Skip if no execution context
        if self._execution_context is None:
            return 1.0

        # Get edge and check for conditions
        edge = self.akg.edges.get((source, target))
        if edge is None or not edge.has_conditions():
            return 1.0

        # Evaluate conditions
        _, boost = edge.evaluate_conditions(self._execution_context)
        return boost

    def _select_next_operator(
        self,
        current: str | None,
        visited: set[str],
        ant: Ant,
    ) -> str | None:
        """Select next operator using ACO probabilistic rule with conditions.

        P(j|i) = (τ_ij^α * η_ij^β * boost_ij * compat_ij) / Σ(τ_ik^α * η_ik^β * boost_ik * compat_ik)

        Level 3: When conditions are enabled, edges with satisfied conditions
        get their probability multiplied by condition_boost.

        Args:
            current: Current operator (None for start)
            visited: Set of already visited operators
            ant: The ant making the selection

        Returns:
            Selected operator ID or None if no valid choice
        """
        valid = self._get_valid_next_operators(current)

        # Filter by energy and avoid some revisits
        candidates = []
        for op_id in valid:
            cost = self._get_energy_cost(op_id)
            if ant.can_continue(cost, self.config.max_steps):
                # Allow revisiting local_search (common in optimization)
                # but limit other revisits
                node = self.akg.get_node(op_id)
                is_local_search = (
                    node and hasattr(node, 'category')
                    and node.category == OperatorCategory.LOCAL_SEARCH
                )
                if op_id not in visited or is_local_search:
                    candidates.append(op_id)

        if not candidates:
            return None

        # Exploration: random choice with small probability
        if random.random() < self.config.exploration_rate:
            return random.choice(candidates)

        # Calculate probabilities using ACO formula with condition boosts
        probabilities = []
        total = 0.0

        for op_id in candidates:
            tau = self._get_pheromone(current, op_id) if current else 1.0
            eta = self._get_heuristic(current, op_id) if current else 1.0

            # Level 3: Get condition boost (1.0 if no conditions)
            boost = self._get_condition_boost(current, op_id) if current else 1.0

            # P(j|i) ∝ τ^α * η^β * boost
            prob = (tau ** self.config.alpha) * (eta ** self.config.beta) * boost
            probabilities.append(prob)
            total += prob

        if total == 0:
            return random.choice(candidates)

        # Normalize and select
        probabilities = [p / total for p in probabilities]

        # Roulette wheel selection
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return candidates[i]

        return candidates[-1]

    def construct_solution(self) -> Ant:
        """Construct a solution (operator sequence) using one ant.

        Supports variable length paths with early stopping.

        Returns:
            Ant with complete path
        """
        # Variable energy: each ant explores different path lengths
        if self.config.variable_energy:
            energy = random.uniform(self.config.energy_min, self.config.energy_max)
        else:
            energy = self.config.initial_energy

        ant = Ant(energy=energy)
        visited: set[str] = set()

        # Must start with a construction operator
        first = self._select_next_operator(None, visited, ant)
        if first is None:
            logger.warning("No valid construction operator found")
            return ant

        cost = self._get_energy_cost(first)
        ant.move_to(first, cost)
        visited.add(first)

        # Continue building sequence
        while True:
            current = ant.path[-1]

            # Check for early stop (after min_steps)
            if (self.config.allow_early_stop and
                len(ant.path) >= self.config.min_steps and
                random.random() < self.config.stop_probability):
                break

            next_op = self._select_next_operator(current, visited, ant)

            if next_op is None:
                break

            cost = self._get_energy_cost(next_op)
            if not ant.can_continue(cost, self.config.max_steps):
                break

            ant.move_to(next_op, cost)
            visited.add(next_op)

        return ant

    def update_pheromones(self, ants: list[Ant]) -> None:
        """Update pheromone levels based on ant solutions.

        Uses MMAS (Min-Max Ant System) if configured:
        - Only best ant deposits pheromone
        - Pheromone bounds enforced
        - Stagnation detection with pheromone reset

        Args:
            ants: List of ants with evaluated fitness
        """
        # Evaporation
        for key in self.pheromones:
            self.pheromones[key] *= (1 - self.config.rho)

        if self.config.use_mmas:
            # MMAS: Only best ant deposits pheromone
            iteration_best = min(
                (ant for ant in ants if ant.fitness is not None),
                key=lambda a: a.fitness,
                default=None
            )

            # Update global best
            if iteration_best and iteration_best.fitness < self.best_fitness:
                self.best_fitness = iteration_best.fitness
                self.best_path = iteration_best.path.copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Choose which best to use for deposit
            if self.config.mmas_global_best and self.best_path:
                best_ant_path = self.best_path
                best_ant_fitness = self.best_fitness
            elif iteration_best:
                best_ant_path = iteration_best.path
                best_ant_fitness = iteration_best.fitness
            else:
                best_ant_path = None
                best_ant_fitness = None

            # Deposit pheromone only on best path
            if best_ant_path and best_ant_fitness and len(best_ant_path) >= 2:
                deposit = self.config.q / best_ant_fitness if best_ant_fitness > 0 else self.config.q

                # Bonus for shorter paths: shorter = more deposit per edge
                if self.config.length_bonus > 0:
                    length_factor = 1.0 + self.config.length_bonus * (self.config.max_steps - len(best_ant_path))
                    deposit *= length_factor

                for i in range(len(best_ant_path) - 1):
                    key = (best_ant_path[i], best_ant_path[i + 1])
                    if key in self.pheromones:
                        self.pheromones[key] += deposit

            # Stagnation: reset pheromones to max if stuck
            if self.stagnation_counter >= self.config.stagnation_limit:
                # Note: Log removed - too noisy in async mode
                for key in self.pheromones:
                    self.pheromones[key] = self.config.max_pheromone
                self.stagnation_counter = 0

        else:
            # Standard ACO: all ants deposit pheromone
            for ant in ants:
                if ant.fitness is None or len(ant.path) < 2:
                    continue

                if ant.fitness > 0:
                    deposit = self.config.q / ant.fitness
                else:
                    deposit = self.config.q

                for i in range(len(ant.path) - 1):
                    key = (ant.path[i], ant.path[i + 1])
                    if key in self.pheromones:
                        self.pheromones[key] += deposit

            # Update best solution
            for ant in ants:
                if ant.fitness is not None and ant.fitness < self.best_fitness:
                    self.best_fitness = ant.fitness
                    self.best_path = ant.path.copy()

        # Enforce pheromone bounds (MMAS requirement)
        for key in self.pheromones:
            self.pheromones[key] = max(self.config.min_pheromone,
                                       min(self.config.max_pheromone, self.pheromones[key]))

    def run_colony(
        self,
        evaluate_fn: Any,
        problem_instance: Any,
    ) -> list[Ant]:
        """Run one iteration of the ant colony.

        Args:
            evaluate_fn: Function to evaluate an operator sequence
            problem_instance: Problem instance for evaluation

        Returns:
            List of ants with evaluated solutions
        """
        self.iterations += 1
        ants = []

        # Each ant constructs a solution
        for _ in range(self.config.n_ants):
            ant = self.construct_solution()
            if ant.path:
                ants.append(ant)

        # Evaluate all solutions
        for ant in ants:
            if ant.path:
                try:
                    ant.fitness = evaluate_fn(ant.path, problem_instance)
                except Exception as e:
                    logger.warning(f"Evaluation failed for path {ant.path}: {e}")
                    ant.fitness = float("inf")

        # Update pheromones
        self.update_pheromones(ants)

        return ants

    def get_best_solution(self) -> tuple[list[str], float]:
        """Get the best solution found so far.

        Returns:
            Tuple of (operator sequence, fitness)
        """
        return self.best_path, self.best_fitness

    def update_pheromones_for_path(self, path: list[str], fitness: float) -> None:
        """Update pheromones for a specific path (for external evaluation).

        This method is used when the engine evaluates algorithms externally
        and wants to update pheromones based on the best solution found.

        Args:
            path: Operator sequence that was evaluated
            fitness: Fitness value (lower is better for minimization)
        """
        if len(path) < 2:
            return

        # Evaporation first
        for key in self.pheromones:
            self.pheromones[key] *= (1 - self.config.rho)

        # Update global best if better
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_path = path.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Deposit pheromone on the path
        deposit = self.config.q / fitness if fitness > 0 else self.config.q

        # Bonus for shorter paths
        if self.config.length_bonus > 0:
            length_factor = 1.0 + self.config.length_bonus * (self.config.max_steps - len(path))
            deposit *= max(1.0, length_factor)

        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in self.pheromones:
                self.pheromones[key] += deposit

        # Stagnation reset (MMAS)
        if self.config.use_mmas and self.stagnation_counter >= self.config.stagnation_limit:
            # Note: Log removed - too noisy in async mode
            for key in self.pheromones:
                self.pheromones[key] = self.config.max_pheromone
            self.stagnation_counter = 0

        # Enforce pheromone bounds
        for key in self.pheromones:
            self.pheromones[key] = max(self.config.min_pheromone,
                                       min(self.config.max_pheromone, self.pheromones[key]))

    def get_pheromone_stats(self) -> dict[str, float]:
        """Get pheromone statistics.

        Returns:
            Dictionary with pheromone stats
        """
        if not self.pheromones:
            return {"min": 0, "max": 0, "mean": 0}

        values = list(self.pheromones.values())
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "n_edges": len(values),
        }


class GreedySelector:
    """Simple greedy selector for comparison.

    Always selects the highest-weight valid transition.
    """

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        max_steps: int = 6,
    ) -> None:
        self.akg = akg
        self.max_steps = max_steps

    def construct_solution(self) -> list[str]:
        """Construct solution using greedy selection.

        Returns:
            Operator sequence
        """
        path = []

        # Start with best construction operator
        construction_ops = self.akg.get_operators_by_category(
            OperatorCategory.CONSTRUCTION
        )
        if not construction_ops:
            return path

        # Pick first construction operator (or best by some metric)
        path.append(construction_ops[0].id)

        # Greedily extend
        while len(path) < self.max_steps:
            current = path[-1]
            valid = self.akg.get_valid_transitions(current)

            if not valid:
                break

            # Select highest weight transition
            best_next = None
            best_weight = -1

            for op_id in valid:
                edge = self.akg.edges.get((current, op_id))
                weight = edge.weight if edge else 0.0
                if weight > best_weight:
                    best_weight = weight
                    best_next = op_id

            if best_next is None:
                break

            path.append(best_next)

        return path


class RandomSelector:
    """Random selector for ablation study.

    Ignores weights, selects randomly from valid transitions.
    Tests value of AKG structure alone (without LLM weights).
    """

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        max_steps: int = 6,
        min_steps: int = 3,
        stop_probability: float = 0.15,
        allow_early_stop: bool = True,
    ) -> None:
        self.akg = akg
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.stop_probability = stop_probability
        self.allow_early_stop = allow_early_stop
        self.best_path: list[str] = []
        self.best_fitness: float = float("inf")

    def construct_solution(self) -> list[str]:
        """Construct solution using random selection.

        Supports variable length paths.

        Returns:
            Operator sequence
        """
        path = []

        # Start with random construction operator
        construction_ops = self.akg.get_operators_by_category(
            OperatorCategory.CONSTRUCTION
        )
        if not construction_ops:
            return path

        path.append(random.choice(construction_ops).id)

        # Randomly extend
        while len(path) < self.max_steps:
            # Check for early stop
            if (self.allow_early_stop and
                len(path) >= self.min_steps and
                random.random() < self.stop_probability):
                break

            current = path[-1]
            valid = self.akg.get_valid_transitions(current)

            if not valid:
                break

            # Random selection (ignores weights)
            path.append(random.choice(valid))

        return path

    def run_batch(
        self,
        n_solutions: int,
        evaluate_fn: Any,
        problem_instance: Any,
    ) -> list[tuple[list[str], float]]:
        """Generate and evaluate multiple random solutions.

        Args:
            n_solutions: Number of solutions to generate
            evaluate_fn: Function to evaluate an operator sequence
            problem_instance: Problem instance for evaluation

        Returns:
            List of (path, fitness) tuples
        """
        results = []

        for _ in range(n_solutions):
            path = self.construct_solution()
            if path:
                try:
                    fitness = evaluate_fn(path, problem_instance)
                    results.append((path, fitness))

                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_path = path.copy()
                except Exception:
                    pass

        return results

    def get_best_solution(self) -> tuple[list[str], float]:
        """Get the best solution found.

        Returns:
            Tuple of (operator sequence, fitness)
        """
        return self.best_path, self.best_fitness


# =============================================================================
# META-GRAPH ACO SELECTOR (Abstract Roles)
# =============================================================================

@dataclass
class MetaAnt:
    """An ant that traverses the MetaGraph using abstract roles."""

    role_path: list[str] = field(default_factory=list)  # AbstractRole values
    operator_path: list[str] = field(default_factory=list)  # Bound concrete operators
    energy: float = 10.0
    fitness: float | None = None

    def can_continue(self, cost: float, max_steps: int = 0) -> bool:
        """Check if ant can continue traversal.

        Path length is determined purely by energy budget.
        max_steps parameter kept for API compatibility but ignored.
        """
        return self.energy >= cost

    def move_to(self, role: str, operator: str, cost: float) -> None:
        """Move to next role, binding to concrete operator."""
        self.role_path.append(role)
        self.operator_path.append(operator)
        self.energy -= cost


@dataclass
class MetaACOConfig(ACOConfig):
    """Configuration for MetaACO traversal.

    Inherits all parameters from ACOConfig and adds:
    - Dynamic energy cost (history-based adjustment)
    - Incompatibility tracking (symbolic reasoning)
    - Dynamic mode operator mode (STATIC/DYNAMIC)
    - Operator selection mode within roles
    """

    # Override: Higher max_steps since energy is the real limit
    max_steps: int = 20

    # ===========================================================================
    # Dynamic Energy Cost (History-Based)
    # ===========================================================================
    # When enabled, energy cost adapts based on operator improvement history:
    # - Operators that improve fitness get lower cost (more energy for productive ops)
    # - Operators that don't improve get higher cost (penalize unproductive ops)
    # This allows ACO to learn which sequences are productive
    use_dynamic_energy: bool = True
    energy_improvement_bonus: float = 0.5  # Cost multiplier when operator improves (0.5 = half cost)
    energy_no_improvement_penalty: float = 1.5  # Cost multiplier when no improvement (1.5 = 50% more)

    # ===========================================================================
    # Incompatibility Tracking (Symbolic Reasoning)
    # ===========================================================================
    # Penalizes transitions that frequently appear in failed paths.
    # Pure symbolic reasoning: no LLM, just frequency counting.
    enable_incompatibility_tracking: bool = True
    incompatibility_failure_threshold: float = 0.3  # >30% of failures = incompatible
    incompatibility_min_samples: int = 10  # Minimum failures before penalizing
    incompatibility_penalty_factor: float = 0.3  # Multiply probability by this

    # ===========================================================================
    # Dynamic mode: Operator Mode
    # ===========================================================================
    # STATIC: Current behavior - uniform random selection within roles (backward compatible)
    # DYNAMIC: ACO pheromones at operator level + optional LLM synthesis
    operator_mode: OperatorMode = OperatorMode.STATIC

    # Only used when operator_mode == DYNAMIC
    enable_synthesis: bool = True  # Enable LLM synthesis when triggered
    operator_alpha: float = 2.0    # Operator pheromone importance (higher = more greedy)
    operator_beta: float = 2.0     # Operator weight importance
    operator_rho: float = 0.1      # Operator pheromone evaporation rate
    operator_tau_min: float = 0.001 # Minimum operator pheromone (lower for more differentiation)
    operator_tau_max: float = 1.0  # Maximum operator pheromone (MMAS normalized)

    # ===========================================================================
    # synthesized Forced Exploration
    # ===========================================================================
    # Forces newly synthesized synthesized operators to be used a minimum number of times
    # before allowing ACO to decide. This ensures synthesized operators get a fair chance.
    synth_forced_exploration: bool = False  # Disabled: use pheromone boost instead
    synth_min_forced_uses: int = 5  # Minimum uses before ACO takes over
    synth_exploration_probability: float = 0.8  # Probability of selecting unexplored synthesized

    # ===========================================================================
    # Operator selection within roles (STATIC mode)
    # ===========================================================================
    # DESIGN PHILOSOPHY (The Binding Loophole):
    # The LLM controls strategy at the ROLE level, not the operator level.
    # When a role has multiple operators (e.g., LS_INTENSIFY_SMALL has [two_opt, swap, insert, invert]),
    # the specific operator is selected according to this mode:
    #   - "uniform": Equiprobable random (DEFAULT, recommended for paper)
    #   - "primary": Always highest priority (deterministic)
    #   - "weighted": Weighted random by operator weights
    # This keeps the LLM as "strategist", not "micro-manager".
    # NOTE: This is only used in STATIC mode. In DYNAMIC mode, operator pheromones guide selection.
    operator_selection_mode: str = "uniform"


class MetaACOSelector:
    """ACO selector for MetaGraph with abstract roles.

    Navigates abstract roles and binds to concrete operators at selection time.
    This enables the same meta-algorithm to work across different domains.

    Key difference from ACOSelector:
    - Nodes are AbstractRoles, not concrete operators
    - At selection time, roles are bound to operators via DomainBindings
    - Returns both role_path and operator_path

    Dynamic mode (DYNAMIC mode):
    - Maintains operator-level pheromones for learning within roles
    - Integrates with multi-agent synthesis system for new operators
    """

    def __init__(
        self,
        instantiated_graph: "InstantiatedGraph",
        config: MetaACOConfig | None = None,
        synthesis_hook: "DynamicSynthesisHook | None" = None,
        role_schema: "RoleSchema | None" = None,
    ) -> None:
        """Initialize MetaACO selector.

        Args:
            instantiated_graph: MetaGraph instantiated for a specific domain
            config: ACO configuration
            synthesis_hook: Optional hook for dynamic operator synthesis (Dynamic mode)
            role_schema: Optional RoleSchema for generalized role queries.
                         If None, uses the graph's schema or defaults to
                         optimization-specific logic.
        """
        from src.geakg.meta_graph import InstantiatedGraph
        from src.geakg.roles import AbstractRole, RoleCategory, ROLE_CATALOG

        self.graph = instantiated_graph
        self.config = config or MetaACOConfig()

        # Use provided schema, or graph's schema, or None (legacy)
        self._role_schema = role_schema or getattr(self.graph.meta_graph, '_role_schema', None)

        # Initialize pheromone matrix for role transitions
        self.pheromones: dict[tuple[str, str], float] = {}
        self._initialize_pheromones()

        # Dynamic mode: Operator-level pheromones (only in DYNAMIC mode)
        self._operator_pheromones: dict[tuple[str, str], float] = {}
        if self.config.operator_mode == OperatorMode.DYNAMIC:
            self._init_operator_pheromones()

        # Dynamic mode: LLM-suggested weights for synthesized operators
        # Maps (role, operator_id) -> weight (0.0-1.0)
        # Used in L2 selection formula: prob = tau^alpha * weight^beta
        self._synthesized_operator_weights: dict[tuple[str, str], float] = {}

        # Dynamic mode: Initial tau for synthesized operators (for protected minimum)
        # Maps (role, operator_id) -> initial_tau
        self._synthesized_operator_initial_tau: dict[tuple[str, str], float] = {}

        # Synthesis hook (Dynamic mode)
        self._synthesis_hook = synthesis_hook

        # Statistics
        self.iterations = 0
        self.best_role_path: list[str] = []
        self.best_operator_path: list[str] = []
        self.best_fitness: float = float("inf")

        # MMAS tracking
        self.stagnation_counter = 0
        self._last_reset_iteration = 0  # Track when last reset happened
        self._reset_cooldown = 50  # Minimum iterations between resets

        # Execution context for conditions
        self._execution_context: "ExecutionContext | None" = None

        # Recent paths for bottleneck analysis (Dynamic mode)
        self._recent_paths: list[list[str]] = []
        self._max_recent_paths: int = 50

        # ===========================================================================
        # Dynamic Energy Cost: Track operator improvement history
        # ===========================================================================
        # Maps operator_id -> (improvements, total_uses)
        # Used to calculate dynamic energy cost based on historical performance
        self._operator_improvement_history: dict[str, tuple[int, int]] = {}

        # ===========================================================================
        # Synthesis Context: Track successful paths and operator stats
        # ===========================================================================
        self._successful_paths: list[dict] = []  # List of {path, fitness, gap}
        self._operator_stats: dict[str, dict] = {}  # operator_id -> {uses, successes, avg_improvement}

        # ===========================================================================
        # Operator Pruning: Remove underperforming operators
        # ===========================================================================
        self._pruned_operators: set[str] = set()  # Operators removed from selection
        self._synth_operator_iteration: dict[str, int] = {}  # When each synthesized operator was added
        self._last_pruning_check: int = 0  # Last iteration when pruning was checked

        # ===========================================================================
        # Incompatibility Tracking (Symbolic Reasoning)
        # ===========================================================================
        self._incompatibility_tracker: IncompatibilityTracker | None = None
        if self.config.enable_incompatibility_tracking:
            self._incompatibility_tracker = IncompatibilityTracker(
                failure_threshold=self.config.incompatibility_failure_threshold,
                min_samples=self.config.incompatibility_min_samples,
                penalty_factor=self.config.incompatibility_penalty_factor,
            )

        # ===========================================================================
        # synthesized Forced Exploration: Track usage of new synthesized operators
        # ===========================================================================
        # Maps (role, operator_id) -> number of times selected
        # Used to force exploration of newly synthesized operators
        self._synth_operator_uses: dict[tuple[str, str], int] = {}

    def _init_operator_pheromones(self) -> None:
        """Initialize pheromones for all operators in all roles (DYNAMIC mode)."""
        for role_value in self.graph.meta_graph.nodes:
            operators = self.graph.bindings.get_operators_for_role(role_value)
            for op in operators:
                self._operator_pheromones[(role_value, op)] = 0.5

        logger.debug(
            f"Initialized {len(self._operator_pheromones)} operator pheromones"
        )

    def set_execution_context(self, context: "ExecutionContext | None") -> None:
        """Set execution context for condition evaluation."""
        self._execution_context = context

    def _initialize_pheromones(self) -> None:
        """Initialize pheromones from edge weights."""
        for (src, tgt), edge in self.graph.meta_graph.edges.items():
            initial = max(self.config.min_pheromone, edge.weight)
            self.pheromones[(src, tgt)] = initial

    def _get_energy_cost(self, role_value: str, operator_id: str | None = None) -> float:
        """Get energy cost for a role, optionally adjusted by operator history.

        Args:
            role_value: The role being executed
            operator_id: Optional operator ID for history-based adjustment

        Returns:
            Energy cost, potentially adjusted by operator's improvement history
        """
        base_cost = 1.0

        # Try schema first, then fall back to ROLE_CATALOG
        if self._role_schema is not None:
            try:
                category = self._role_schema.get_role_category(role_value)
                base_cost = self.config.energy_costs.get(category, 1.0)
            except KeyError:
                pass
        else:
            from src.geakg.roles import ROLE_CATALOG, AbstractRole
            try:
                role = AbstractRole(role_value)
                category = ROLE_CATALOG[role]["category"].value
                base_cost = self.config.energy_costs.get(category, 1.0)
            except (ValueError, KeyError):
                pass

        # Apply dynamic cost adjustment based on operator history
        if self.config.use_dynamic_energy and operator_id:
            history = self._operator_improvement_history.get(operator_id)
            if history and history[1] >= 3:  # Need at least 3 uses for reliable estimate
                improvements, total = history
                improvement_rate = improvements / total

                if improvement_rate > 0.5:
                    # Operator usually improves - lower cost (more energy for productive ops)
                    base_cost *= self.config.energy_improvement_bonus
                elif improvement_rate < 0.2:
                    # Operator rarely improves - higher cost (penalize unproductive ops)
                    base_cost *= self.config.energy_no_improvement_penalty

        return base_cost

    def update_operator_improvement(self, operator_id: str, improved: bool) -> None:
        """Update improvement history for an operator.

        Args:
            operator_id: The operator that was executed
            improved: True if the operator improved the solution
        """
        if operator_id not in self._operator_improvement_history:
            self._operator_improvement_history[operator_id] = (0, 0)

        improvements, total = self._operator_improvement_history[operator_id]
        if improved:
            improvements += 1
        total += 1
        self._operator_improvement_history[operator_id] = (improvements, total)

    def _get_valid_next_roles(self, current: str | None) -> list[str]:
        """Get valid next roles from current position."""
        if current is None:
            # Start with entry roles
            return self.graph.get_entry_roles()

        return self.graph.get_successors(current)

    def _get_condition_boost(self, source: str, target: str) -> float:
        """Get condition boost for a role transition."""
        if not self.config.enable_conditions or self._execution_context is None:
            return 1.0

        try:
            edge = self.graph.get_edge(source, target)

            if edge is None or not edge.has_conditions():
                return 1.0

            effective_weight = edge.get_effective_weight(self._execution_context)
            if effective_weight > edge.weight:
                return edge.condition_boost

            return 1.0

        except (ValueError, KeyError):
            return 1.0

    def _get_incompatibility_penalty(self, source: str | None, target: str) -> float:
        """Get incompatibility penalty for a role transition.

        Args:
            source: Source role (None for start).
            target: Target role.

        Returns:
            Penalty multiplier (0.3-1.0). 1.0 means no penalty.
        """
        if source is None or self._incompatibility_tracker is None:
            return 1.0
        return self._incompatibility_tracker.get_penalty(source, target)

    def _select_next_role(
        self,
        current: str | None,
        visited: set[str],
        ant: MetaAnt,
    ) -> str | None:
        """Select next role using ACO probabilistic rule.

        Path termination is controlled by energy budget.

        P(j|i) = (τ_ij^α * η_ij^β * boost_ij * compat_ij) / Σ(...)

        where compat_ij is the incompatibility penalty (1.0 if compatible,
        penalty_factor if frequently in failed paths).
        """
        valid = self._get_valid_next_roles(current)

        # Build candidates list
        candidates = []
        for role_value in valid:
            # Allow revisiting roles in revisitable categories
            is_revisitable = False
            if self._role_schema is not None:
                try:
                    cat = self._role_schema.get_role_category(role_value)
                    is_revisitable = self._role_schema.is_revisitable_category(cat)
                except KeyError:
                    pass
            else:
                # Legacy: hardcoded check for optimization roles
                is_revisitable = (
                    role_value.startswith("ls_") or
                    role_value.startswith("pert_")
                )
            if role_value not in visited or is_revisitable:
                candidates.append(role_value)

        if not candidates:
            return None

        # Exploration
        if random.random() < self.config.exploration_rate:
            return random.choice(candidates)

        # Calculate probabilities
        probabilities = []
        total = 0.0

        for role_value in candidates:
            tau = self.pheromones.get((current, role_value), self.config.min_pheromone) if current else 1.0

            # Get edge weight as heuristic
            if current:
                edge = self.graph.get_edge(current, role_value)
                eta = edge.weight if edge else 0.01
            else:
                eta = 1.0

            # Condition boost
            boost = self._get_condition_boost(current, role_value) if current else 1.0

            # Incompatibility penalty (symbolic reasoning)
            compat = self._get_incompatibility_penalty(current, role_value)

            prob = (tau ** self.config.alpha) * (eta ** self.config.beta) * boost * compat
            probabilities.append(prob)
            total += prob

        if total == 0:
            return random.choice(candidates)

        # Normalize and select
        probabilities = [p / total for p in probabilities]

        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return candidates[i]

        return candidates[-1]

    def _select_operator_for_role(
        self,
        role: "str | AbstractRole",
        problem_size: int | None = None,
    ) -> str:
        """Select operator for a role based on operating mode.

        STATIC mode: Uses uniform/primary/weighted selection from bindings.
        DYNAMIC mode: Uses ACO with operator pheromones.

        Args:
            role: The abstract role to select operator for (str or AbstractRole).
            problem_size: Optional problem size hint.

        Returns:
            Selected operator ID.
        """
        from src.geakg.roles import AbstractRole

        # Normalize to string for lookups
        role_str = role.value if isinstance(role, AbstractRole) else role

        # Get available operators for this role (predefined bindings)
        operators = list(self.graph.bindings.get_operators_for_role(role))

        # In DYNAMIC mode, also include synthesized operators
        if self.config.operator_mode == OperatorMode.DYNAMIC:
            synthesized = [
                op for (r, op) in self._operator_pheromones.keys()
                if r == role_str and op not in operators
            ]
            operators.extend(synthesized)

        # Filter out disabled operators (those that failed repeatedly)
        from src.geakg.execution import get_disabled_operators
        disabled = get_disabled_operators()
        operators = [op for op in operators if op not in disabled]

        # Filter out pruned operators (underperforming)
        operators = [op for op in operators if op not in self._pruned_operators]

        if not operators:
            raise ValueError(f"No operators for role {role_str}")

        # STATIC MODE: Use original selection logic
        if self.config.operator_mode == OperatorMode.STATIC:
            return self.graph.select_operator(
                role,
                problem_size,
                mode=self.config.operator_selection_mode,
            )

        # DYNAMIC MODE: ACO selection with operator pheromones

        # ===========================================================================
        # synthesized Forced Exploration: Prioritize unexplored synthesized operators
        # Higher effect_size operators get more forced uses
        # ===========================================================================
        if self.config.synth_forced_exploration:
            # Find synthesized operators for this role that haven't been used enough
            unexplored_synth = []
            for op in operators:
                key = (role.value, op)
                if key in self._synth_operator_uses:
                    current_uses, required_uses = self._synth_operator_uses[key]
                    if current_uses < required_uses:
                        unexplored_synth.append((op, required_uses - current_uses))

            # Prioritize operators with more remaining required uses (higher effect_size)
            if unexplored_synth:
                # Sort by remaining uses descending (high effect_size first)
                unexplored_synth.sort(key=lambda x: x[1], reverse=True)

                if random.random() < self.config.synth_exploration_probability:
                    # Select from top candidates (same remaining uses = same priority)
                    top_remaining = unexplored_synth[0][1]
                    top_candidates = [op for op, rem in unexplored_synth if rem == top_remaining]
                    selected = random.choice(top_candidates)

                    # Increment usage counter
                    key = (role.value, selected)
                    current, required = self._synth_operator_uses[key]
                    self._synth_operator_uses[key] = (current + 1, required)

                    logger.debug(
                        f"[synthesized-EXPLORE] Forced selection of {selected} in {role.value} "
                        f"(use {current + 1}/{required})"
                    )
                    return selected

        # Standard ACO selection with operator pheromones
        probs = []
        for op in operators:
            tau = self._operator_pheromones.get((role.value, op), 1.0)
            # Get operator weight: check synthesized weights first (synthesized),
            # then fall back to bindings (predefined operators)
            key = (role.value, op)
            if key in self._synthesized_operator_weights:
                # synthesized synthesized operator: use LLM-suggested weight
                weight = self._synthesized_operator_weights[key]
            else:
                # Predefined operator: use weight from bindings
                weight = self.graph.bindings.get_operator_weight(role, op)
            prob = (tau ** self.config.operator_alpha) * (weight ** self.config.operator_beta)
            probs.append(prob)

        total = sum(probs)
        if total == 0:
            return random.choice(operators)

        probs = [p / total for p in probs]
        selected = random.choices(operators, weights=probs, k=1)[0]

        # Track synthesized operator usage even when selected by ACO (not forced)
        key = (role.value, selected)
        if key in self._synth_operator_uses:
            current, required = self._synth_operator_uses[key]
            self._synth_operator_uses[key] = (current + 1, required)

        return selected

    def construct_solution(self, problem_size: int | None = None) -> MetaAnt:
        """Construct a solution using one ant.

        Args:
            problem_size: Optional problem size for operator selection

        Returns:
            MetaAnt with role_path and operator_path
        """
        # Variable energy: each ant explores different path lengths
        if self.config.variable_energy:
            energy = random.uniform(self.config.energy_min, self.config.energy_max)
        else:
            energy = self.config.initial_energy

        ant = MetaAnt(energy=energy)
        visited: set[str] = set()

        # Start with entry role
        first_role = self._select_next_role(None, visited, ant)
        if first_role is None:
            logger.warning("No valid entry role found")
            return ant

        # Bind role to operator (using mode-dependent selection)
        try:
            first_operator = self._select_operator_for_role(
                first_role,
                problem_size,
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to bind role {first_role}: {e}")
            return ant

        # Get cost with operator history adjustment
        cost = self._get_energy_cost(first_role, first_operator)
        ant.move_to(first_role, first_operator, cost)
        visited.add(first_role)

        # Continue building sequence
        # Path terminates when energy runs out or no valid next role
        while True:
            current_role = ant.role_path[-1]

            next_role = self._select_next_role(current_role, visited, ant)
            if next_role is None:
                break

            # Bind role to operator (using mode-dependent selection)
            try:
                next_operator = self._select_operator_for_role(
                    next_role,
                    problem_size,
                )
            except (ValueError, KeyError):
                break

            # Check energy with operator-specific cost adjustment
            cost = self._get_energy_cost(next_role, next_operator)
            if not ant.can_continue(cost):
                break

            ant.move_to(next_role, next_operator, cost)
            visited.add(next_role)

        # Track for bottleneck analysis (Dynamic mode)
        if ant.role_path:
            self._recent_paths.append(ant.role_path.copy())
            if len(self._recent_paths) > self._max_recent_paths:
                self._recent_paths.pop(0)

        return ant

    def update_operator_pheromones(
        self,
        ant: MetaAnt,
        fitness: float,
    ) -> None:
        """Update pheromones for operators (DYNAMIC mode only).

        Args:
            ant: Ant with role_path and operator_path.
            fitness: Fitness value (lower is better).
        """
        if self.config.operator_mode != OperatorMode.DYNAMIC:
            return  # No-op in STATIC mode

        # Evaporate all operator pheromones
        rho = self.config.operator_rho
        for key in self._operator_pheromones:
            self._operator_pheromones[key] *= (1 - rho)
            # synthesized operators: protected minimum = 50% of initial tau
            # This prevents synthesized operators from losing their advantage
            if key in self._synthesized_operator_initial_tau:
                tau_min = self._synthesized_operator_initial_tau[key] * 0.5
                tau_min = max(tau_min, self.config.operator_tau_min)
            elif key[1].endswith('_base'):
                # _base operators: guaranteed fallback minimum of 0.1
                # This ensures each role always has a functional operator
                tau_min = 0.1
            else:
                tau_min = self.config.operator_tau_min
            self._operator_pheromones[key] = max(
                self._operator_pheromones[key],
                tau_min,
            )

        # Deposit on best ant's operator path
        if not ant.role_path or not ant.operator_path:
            return

        deposit = self.config.q / fitness if fitness > 0 else self.config.q

        for role, op in zip(ant.role_path, ant.operator_path):
            key = (role, op)
            if key in self._operator_pheromones:
                self._operator_pheromones[key] += deposit
                self._operator_pheromones[key] = min(
                    self._operator_pheromones[key],
                    self.config.operator_tau_max,
                )

    def register_new_operator(
        self,
        role: str,
        operator_id: str,
        initial_tau: float | None = None,
        weight: float | None = None,
        effect_size: float | None = None,
    ) -> None:
        """Initialize pheromone for newly synthesized operator (DYNAMIC mode).

        Args:
            role: Role value the operator is added to.
            operator_id: ID of the new operator.
            initial_tau: Initial pheromone value (if None, uses average of role).
            weight: LLM-suggested L2 weight (0.0-1.0). If None, defaults to 0.5.
            effect_size: Validation effect size. Higher = more forced exploration.
        """
        if self.config.operator_mode != OperatorMode.DYNAMIC:
            return  # No-op in STATIC mode

        # Use provided tau or default to max (LLM operators deserve a fair chance)
        if initial_tau is None:
            initial_tau = self.config.operator_tau_max  # Start at max (1.0)

        self._operator_pheromones[(role, operator_id)] = initial_tau

        # Store initial tau for protected minimum (synthesized operators don't fall below 50% of initial)
        self._synthesized_operator_initial_tau[(role, operator_id)] = initial_tau

        # Store LLM-suggested weight for L2 selection
        if weight is None:
            weight = 0.5  # Default moderate weight
        self._synthesized_operator_weights[(role, operator_id)] = weight

        # Track when this synthesized operator was added (for pruning grace period)
        self._synth_operator_iteration[operator_id] = self.iterations

        # Boost initial pheromone based on effect size (instead of forced executions)
        # Higher effect size = higher initial pheromone = more likely to be selected by ACO
        if effect_size is not None and effect_size > 0:
            if effect_size > 2.0:
                # Exceptional operator: 3x pheromone boost
                boost = 3.0
            elif effect_size > 1.0:
                # Very good operator: 2x pheromone boost
                boost = 2.0
            elif effect_size > 0.5:
                # Good operator: 1.5x pheromone boost
                boost = 1.5
            else:
                boost = 1.0
            self._operator_pheromones[(role, operator_id)] = initial_tau * boost
            logger.info(
                f"[DYNAMIC] Registered operator {operator_id} in {role} "
                f"with τ={initial_tau * boost:.3f} (boosted {boost}x), weight={weight:.2f}, "
                f"effect_size={effect_size:.2f}"
            )
        else:
            logger.info(
                f"[DYNAMIC] Registered operator {operator_id} in {role} "
                f"with τ={initial_tau:.3f}, weight={weight:.2f}"
            )

    def get_operator_pheromones(self) -> dict[tuple[str, str], float]:
        """Get the operator pheromone matrix.

        Returns:
            Dictionary mapping (role, operator) to pheromone value.
        """
        return self._operator_pheromones.copy()

    def transfer_pheromones_from(
        self,
        role_pheromones: dict[tuple[str, str], float],
        operator_pheromones: dict[tuple[str, str], float],
    ) -> None:
        """Transfer pheromones from a previous selector.

        Useful for maintaining learned knowledge across refinement rounds.
        Only transfers values for edges/operators that exist in this selector.
        New operators retain their initial pheromone values.

        Args:
            role_pheromones: Role transition pheromones from previous selector
            operator_pheromones: Operator pheromones from previous selector
        """
        # Transfer role-level pheromones
        transferred_roles = 0
        for key, tau in role_pheromones.items():
            if key in self.pheromones:
                self.pheromones[key] = tau
                transferred_roles += 1

        # Transfer operator-level pheromones
        transferred_ops = 0
        for key, tau in operator_pheromones.items():
            if key in self._operator_pheromones:
                self._operator_pheromones[key] = tau
                transferred_ops += 1

        logger.debug(
            f"Transferred pheromones: {transferred_roles} role edges, "
            f"{transferred_ops} operators"
        )

    def get_synth_exploration_status(self) -> dict[str, Any]:
        """Get synthesized forced exploration status.

        Returns:
            Dictionary with exploration status for each synthesized operator.
        """
        status = {}

        for (role, op_id), (current, required) in self._synth_operator_uses.items():
            status[f"{role}:{op_id}"] = {
                "uses": current,
                "required": required,
                "remaining": max(0, required - current),
                "exploration_complete": current >= required,
            }

        return status

    def record_path_outcome(
        self,
        role_path: list[str],
        is_failure: bool,
    ) -> None:
        """Record path outcome for incompatibility tracking.

        Args:
            role_path: Sequence of role values.
            is_failure: True if the path resulted in a poor solution.
        """
        if self._incompatibility_tracker is not None:
            self._incompatibility_tracker.record_path(role_path, is_failure)

    def get_incompatibility_stats(self) -> dict[str, Any]:
        """Get incompatibility tracking statistics.

        Returns:
            Dictionary with tracking stats, or empty dict if disabled.
        """
        if self._incompatibility_tracker is None:
            return {}
        return self._incompatibility_tracker.get_stats()

    def get_incompatible_transitions(self) -> set[tuple[str, str]]:
        """Get all currently incompatible transitions.

        Returns:
            Set of (source, target) tuples that are penalized.
        """
        if self._incompatibility_tracker is None:
            return set()
        return self._incompatibility_tracker.get_incompatible_transitions()

    def update_pheromones_for_path(
        self,
        role_path: list[str],
        fitness: float,
        operator_path: list[str] | None = None,
    ) -> None:
        """Update pheromones for a role path.

        Args:
            role_path: Sequence of role values
            fitness: Fitness value (lower is better)
            operator_path: Sequence of operator names (optional)
        """
        if len(role_path) < 2:
            return

        # Evaporation
        for key in self.pheromones:
            self.pheromones[key] *= (1 - self.config.rho)

        # Update best
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_role_path = role_path.copy()
            if operator_path is not None:
                self.best_operator_path = operator_path.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Deposit
        deposit = self.config.q / fitness if fitness > 0 else self.config.q

        for i in range(len(role_path) - 1):
            key = (role_path[i], role_path[i + 1])
            if key in self.pheromones:
                self.pheromones[key] += deposit

        # Stagnation reset (MMAS) with cooldown to avoid constant resets
        iterations_since_reset = self.iterations - self._last_reset_iteration
        if (self.config.use_mmas and
            self.stagnation_counter >= self.config.stagnation_limit and
            iterations_since_reset >= self._reset_cooldown):

            # Reset edge pheromones
            for key in self.pheromones:
                self.pheromones[key] = self.config.max_pheromone

            # Reset operator pheromones to allow exploration
            for key in self._operator_pheromones:
                role, op_id = key
                # LLM-synthesized operators: reset to their initial tau (protected at 1.0)
                if key in self._synthesized_operator_initial_tau:
                    self._operator_pheromones[key] = self._synthesized_operator_initial_tau[key]
                else:
                    # Base/generic operators: reset to 0.5 (must prove their value)
                    self._operator_pheromones[key] = 0.5

            self.stagnation_counter = 0
            self._last_reset_iteration = self.iterations

        # Bounds
        for key in self.pheromones:
            self.pheromones[key] = max(
                self.config.min_pheromone,
                min(self.config.max_pheromone, self.pheromones[key])
            )

    def get_best_solution(self) -> tuple[list[str], list[str], float]:
        """Get the best solution found.

        Returns:
            Tuple of (role_path, operator_path, fitness)
        """
        return self.best_role_path, self.best_operator_path, self.best_fitness

    # =========================================================================
    # synthesized Synthesis Context: Track operator stats and successful paths
    # =========================================================================

    def record_operator_result(
        self,
        operator_id: str,
        is_synth: bool,
        fitness_before: float,
        fitness_after: float,
    ) -> None:
        """Record result of applying an operator for synthesized synthesis context.

        Args:
            operator_id: The operator that was applied.
            is_synth: True if this is an synthesized-synthesized operator.
            fitness_before: Fitness before applying the operator.
            fitness_after: Fitness after applying the operator.
        """
        if operator_id not in self._operator_stats:
            self._operator_stats[operator_id] = {
                "operator_id": operator_id,
                "is_synth": is_synth,
                "total_uses": 0,
                "total_improvement": 0.0,
            }

        stats = self._operator_stats[operator_id]
        stats["total_uses"] += 1

        # Record relative improvement if fitness improved
        improved = fitness_before > 0 and fitness_after < fitness_before
        if improved:
            relative_improvement = (fitness_before - fitness_after) / fitness_before
            stats["total_improvement"] += relative_improvement

        # Also update improvement history for pruning
        self.update_operator_improvement(operator_id, improved)

    def record_successful_path(
        self,
        role_path: list[str],
        operator_path: list[str],
    ) -> None:
        """Record a path that improved fitness.

        Args:
            role_path: List of role names in the path.
            operator_path: List of operator names in the path.
        """
        self._successful_paths.append({
            "roles": role_path.copy(),
            "operators": operator_path.copy(),
        })

    def get_successful_paths(self) -> list:
        """Get all recorded successful paths.

        Returns:
            List of SuccessfulPath objects.
        """
        return self._successful_paths.copy()

    def get_operator_stats(self) -> dict:
        """Get operator statistics for synthesized synthesis context.

        Returns:
            Dict mapping operator_id to OperatorStats.
        """
        return self._operator_stats.copy()

    # =========================================================================
    # Operator Pruning: Remove underperforming operators
    # =========================================================================

    def check_and_prune_operators(self, current_iteration: int) -> list[str]:
        """Check and prune underperforming operators.

        Removes operators that:
        1. Have been used enough times (min_uses)
        2. Have low improvement rate (< threshold)
        3. Have low pheromone relative to max (< threshold)

        synthesized operators have a grace period before they can be pruned.

        Args:
            current_iteration: Current optimization iteration.

        Returns:
            List of operator IDs that were pruned.
        """
        if not self.config.enable_pruning:
            return []

        # Only check at intervals
        if current_iteration - self._last_pruning_check < self.config.pruning_check_interval:
            return []

        self._last_pruning_check = current_iteration
        pruned = []

        # Get max pheromone for relative threshold
        max_pheromone = max(self._operator_pheromones.values()) if self._operator_pheromones else 1.0
        pheromone_threshold = max_pheromone * self.config.pruning_pheromone_threshold

        # Count active (non-pruned) operators per role to avoid removing the last one
        active_ops_per_role: dict[str, int] = {}
        for (role, op_id), _ in self._operator_pheromones.items():
            if op_id not in self._pruned_operators:
                active_ops_per_role[role] = active_ops_per_role.get(role, 0) + 1

        for (role, op_id), tau in list(self._operator_pheromones.items()):
            # Skip already pruned operators
            if op_id in self._pruned_operators:
                continue

            # CRITICAL: Never prune the last operator in a role
            if active_ops_per_role.get(role, 0) <= 1:
                continue

            # Skip perturbation/exploration operators - they intentionally worsen solutions
            # Use schema if available, otherwise fall back to prefix check
            is_perturbation = False
            if self._role_schema is not None:
                try:
                    cat = self._role_schema.get_role_category(role)
                    is_perturbation = cat in ("perturbation",)
                except KeyError:
                    pass
            else:
                is_perturbation = op_id.startswith("pert_") or role.startswith("pert_")
            if is_perturbation:
                continue

            # Check if operator has enough usage data
            history = self._operator_improvement_history.get(op_id)
            if not history or history[1] < self.config.pruning_min_uses:
                continue

            improvements, total_uses = history
            improvement_rate = improvements / total_uses if total_uses > 0 else 0

            # Check synthesized grace period
            is_synth = op_id in self._synth_operator_iteration
            if is_synth:
                synth_iteration = self._synth_operator_iteration[op_id]
                if current_iteration - synth_iteration < self.config.pruning_grace_period:
                    continue  # Still in grace period

            # Pruning criteria: low improvement AND low pheromone
            should_prune = (
                improvement_rate < self.config.pruning_improvement_threshold
                and tau < pheromone_threshold
            )

            if should_prune:
                self._pruned_operators.add(op_id)
                pruned.append(op_id)
                # Update active count for this role
                active_ops_per_role[role] = active_ops_per_role.get(role, 1) - 1
                logger.info(
                    f"[PRUNING] Removed operator {op_id}: "
                    f"improvement_rate={improvement_rate:.1%}, "
                    f"pheromone={tau:.3f} (threshold={pheromone_threshold:.3f}), "
                    f"uses={total_uses}"
                )

        if pruned:
            logger.info(f"[PRUNING] Pruned {len(pruned)} operators at iteration {current_iteration}")

        return pruned

    def get_pruned_operators(self) -> set[str]:
        """Get set of pruned operator IDs.

        Returns:
            Set of operator IDs that have been pruned.
        """
        return self._pruned_operators.copy()

    def is_operator_pruned(self, operator_id: str) -> bool:
        """Check if an operator has been pruned.

        Args:
            operator_id: The operator ID to check.

        Returns:
            True if the operator has been pruned.
        """
        return operator_id in self._pruned_operators
