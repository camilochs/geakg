"""NAS Symbolic Executor: traverse the MetaGraph using learned pheromones.

Analogous to SymbolicExecutor (Case Study 2, TSP/JSSP/QAP) but adapted
for NAS (Case Study 1).  Supports both Graph (NAS-Bench-Graph) and Cell
(NAS-Bench-201) architectures via pluggable factory/validator callables.

The executor implements the same 3 mechanisms as the Caso 2 executor:

1. **Iterative refinement**: maintain current + best architecture, evaluate
   after EACH operator, accept/reject greedily (not single-shot walks).
2. **Stagnation detection + perturbation**: after N steps without improvement
   force reg_* roles (structural perturbation) to escape local optima.
3. **Intelligent restarts**: restart from best_arch + strong mutation instead
   of random architectures to preserve good sub-structures.

Role categories map to search phases:
  - topo_* / act_* = INTENSIFICATION (fine refinement)
  - train_*        = INTENSIFICATION escalada (joint mutations)
  - reg_*          = PERTURBATION (structural changes: shuffle, reset)
  - eval_*         = noop, skipped

This demonstrates that the GEAKG captures autonomous, executable
procedural knowledge: given only a pheromone snapshot + base operators,
it generates good architectures with 0 LLM tokens and 0 ACO training.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from src.geakg.safe_exec import safe_call_operator

from src.domains.nas.graph_architecture import (
    GraphArchitecture,
    GRAPH_NUM_NODES,
    GRAPH_NUM_OPS,
    GRAPH_MAX_CONN,
)
from src.domains.nas.cell_architecture import (
    CellArchitecture,
    NUM_EDGES as CELL_NUM_EDGES,
    NUM_OPS as CELL_NUM_OPS,
)


@dataclass
class NASExecutionResult:
    """Result of NAS symbolic execution."""

    best_architecture: Any  # GraphArchitecture or CellArchitecture
    best_accuracy: float
    total_evals: int
    total_walks: int
    elapsed_time: float
    convergence: list[float] = field(default_factory=list)
    walk_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_architecture": (
                self.best_architecture.to_dict()
                if self.best_architecture
                else None
            ),
            "best_accuracy": self.best_accuracy,
            "total_evals": self.total_evals,
            "total_walks": self.total_walks,
            "elapsed_time_s": self.elapsed_time,
            "convergence": self.convergence,
        }


@dataclass
class NASSymbolicExecutor:
    """Symbolic executor for NAS: traverse MetaGraph using learned pheromones.

    Given a pheromone snapshot (edge weights from ACO training) and a set
    of operators indexed by role, this executor generates architectures by
    iterative refinement guided by pheromones.

    Key mechanisms (mirroring Case Study 2 SymbolicExecutor):
    1. Iterative refinement: eval per step, accept/reject greedily.
    2. Stagnation detection → force reg_* (perturbation) roles.
    3. Intelligent restarts from best_arch + strong mutation.

    No ACO loop, no LLM calls — pure symbolic execution of the KG.
    """

    # Pheromones: {"src_role->tgt_role": float}
    pheromones: dict[str, float]

    # Operators by role: {"topo_feedforward": [callable, ...]}
    operators_by_role: dict[str, list]

    # Execution parameters
    n_restarts: int = 5
    stagnation_threshold: int = 8  # steps without improvement → perturb
    temperature: float = 0.5  # softmax temperature (0=greedy, ∞=uniform)
    alpha: float = 2.0  # pheromone exponent (τ^α)
    restart_from_best_prob: float = 0.7  # probability of restarting from best
    seed: int = 42

    # Architecture factory/validator — defaults to GraphArchitecture (backwards compat).
    # Set these to CellArchitecture equivalents when using NAS-Bench-201.
    arch_random: Callable[..., Any] | None = None  # (rng) -> arch
    arch_validator: Callable[[Any], bool] | None = None  # (arch) -> bool
    arch_perturb_fallback: Callable[..., Any] | None = None  # (arch, rng) -> arch

    def execute(
        self,
        evaluator: Any,
        n_evals_budget: int = 200,
    ) -> NASExecutionResult:
        """Run symbolic execution with iterative refinement.

        Unlike the old single-shot walk approach, this method:
        - Maintains current + best architecture across steps
        - Evaluates after EACH operator application (not at end)
        - Detects stagnation and forces perturbation roles
        - Restarts from best_arch with mutation (not random)

        Args:
            evaluator: NASBenchGraphEvaluator with .evaluate(arch) -> float.
            n_evals_budget: Maximum number of architecture evaluations.

        Returns:
            NASExecutionResult with best architecture found.
        """
        rng = random.Random(self.seed)
        t0 = time.time()

        # Parse adjacency from pheromone keys
        adjacency = self._build_adjacency()

        # Identify perturbation roles (reg_*) and skip roles (eval_*)
        perturb_roles = {
            r for r in self.operators_by_role if r.startswith("reg_")
        }
        skip_roles = {
            r for r in self.operators_by_role if r.startswith("eval_")
        }

        best_acc = -1.0
        best_arch: Any = None
        convergence: list[float] = []
        walk_details: list[dict[str, Any]] = []
        eval_count = 0
        restart_count = 0

        # Budget per restart (leave room for all restarts)
        budget_per_restart = max(
            10, n_evals_budget // max(1, self.n_restarts),
        )

        for restart in range(self.n_restarts):
            if eval_count >= n_evals_budget:
                break

            restart_count += 1

            # --- Intelligent restart (Mejora 3) ---
            if best_arch is not None and rng.random() < self.restart_from_best_prob:
                # Restart from best with strong mutation via reg_* operator
                current = best_arch.copy()
                applied_perturb = False
                for role in perturb_roles:
                    if role in self.operators_by_role:
                        ops = self.operators_by_role[role]
                        op = rng.choice(ops)
                        new_arch = safe_call_operator(op, current, None)
                        if new_arch is not None and self._validate_arch(new_arch):
                            current = new_arch
                            applied_perturb = True
                            break
                # If no perturbation succeeded, apply fallback reset
                if not applied_perturb:
                    if self.arch_perturb_fallback is not None:
                        current = self.arch_perturb_fallback(best_arch, rng)
                    else:
                        # Default: Graph fallback (connectivity reset)
                        current = best_arch.copy()
                        current.connectivity = [
                            rng.randint(0, GRAPH_MAX_CONN - 1)
                            for _ in range(GRAPH_NUM_NODES)
                        ]
            else:
                if self.arch_random is not None:
                    current = self.arch_random(rng)
                else:
                    current = GraphArchitecture.random(rng)

            # Evaluate initial architecture for this restart
            current_acc = evaluator.evaluate(current)
            eval_count += 1

            if current_acc > best_acc:
                best_acc = current_acc
                best_arch = current.copy()
            convergence.append(best_acc)

            # --- Iterative refinement loop (Mejora 1) ---
            stagnation = 0
            current_role = self._select_entry_role(adjacency, rng)
            if not current_role:
                continue

            step_count = 0
            while eval_count < n_evals_budget and step_count < budget_per_restart:
                # --- Stagnation detection (Mejora 2) ---
                if stagnation >= self.stagnation_threshold:
                    # Force a perturbation role to escape local optimum
                    candidates = [
                        r for r in perturb_roles
                        if r in self.operators_by_role
                    ]
                    if candidates:
                        current_role = rng.choice(candidates)
                    stagnation = 0

                # Skip eval_* roles (noop operators waste budget)
                if current_role in skip_roles:
                    next_role = self._select_next_role(
                        current_role, adjacency, rng,
                    )
                    current_role = (
                        next_role
                        if next_role is not None
                        else self._select_entry_role(adjacency, rng)
                    )
                    continue

                # Select operator by pheromone within current role
                if current_role not in self.operators_by_role:
                    next_role = self._select_next_role(
                        current_role, adjacency, rng,
                    )
                    current_role = (
                        next_role
                        if next_role is not None
                        else self._select_entry_role(adjacency, rng)
                    )
                    continue

                op = self._select_operator_by_pheromone(
                    current_role, rng,
                )

                # Apply operator (with timeout for LLM-generated code)
                new_arch = safe_call_operator(op, current, None)
                if new_arch is None or not self._validate_arch(new_arch):
                    stagnation += 1
                    next_role = self._select_next_role(
                        current_role, adjacency, rng,
                    )
                    current_role = (
                        next_role
                        if next_role is not None
                        else self._select_entry_role(adjacency, rng)
                    )
                    continue

                # Evaluate (1 eval per step)
                new_acc = evaluator.evaluate(new_arch)
                eval_count += 1
                step_count += 1

                # --- Acceptance criteria (like Case 2 lines 220-247) ---
                is_perturb_role = current_role in perturb_roles

                if new_acc > best_acc:
                    # Global improvement: always accept
                    best_acc = new_acc
                    best_arch = new_arch.copy()
                    current = new_arch
                    current_acc = new_acc
                    stagnation = 0
                elif new_acc > current_acc:
                    # Local improvement: accept
                    current = new_arch
                    current_acc = new_acc
                    stagnation = 0
                elif is_perturb_role:
                    # In perturbation: accept any result (escape)
                    current = new_arch
                    current_acc = new_acc
                    # Don't reset stagnation — we forced this
                else:
                    stagnation += 1

                convergence.append(best_acc)

                walk_details.append({
                    "restart": restart,
                    "step": step_count,
                    "accuracy": new_acc,
                    "role": current_role,
                    "stagnation": stagnation,
                    "perturb": is_perturb_role,
                })

                # Select next role by pheromones
                next_role = self._select_next_role(
                    current_role, adjacency, rng,
                )
                if next_role is None:
                    current_role = self._select_entry_role(adjacency, rng)
                else:
                    current_role = next_role

        elapsed = time.time() - t0

        logger.info(
            f"[NASSymbolicExecutor] Done: {eval_count} evals, "
            f"{restart_count} restarts, best_acc={best_acc:.2f}%, "
            f"{elapsed:.2f}s"
        )

        return NASExecutionResult(
            best_architecture=best_arch,
            best_accuracy=best_acc,
            total_evals=eval_count,
            total_walks=restart_count,
            elapsed_time=elapsed,
            convergence=convergence,
            walk_details=walk_details,
        )

    def _select_operator_by_pheromone(
        self,
        role: str,
        rng: random.Random,
    ) -> Any:
        """Select operator within a role using pheromone-weighted probabilities.

        Uses role_frequency from pheromone snapshot as proxy for operator
        quality. Operators with higher incoming pheromone weight for their
        role get selected more often, mimicking ACO's τ^α selection.

        Falls back to uniform random if no pheromone signal available.
        """
        ops = self.operators_by_role[role]
        if len(ops) == 1:
            return ops[0]

        # Compute weight for each operator based on pheromone signal
        # Use the incoming pheromone to this role as a base weight,
        # then differentiate operators by their individual weight field
        weights = []
        for op in ops:
            base_weight = getattr(op, "weight", 1.0)
            # Check if there's a specific pheromone for this operator
            op_id = getattr(op, "operator_id", "")
            op_tau = self.pheromones.get(f"{role}:{op_id}", base_weight)
            weights.append(max(0.01, op_tau) ** self.alpha)

        return self._weighted_choice_ops(ops, weights, rng)

    def _weighted_choice_ops(
        self,
        items: list,
        weights: list[float],
        rng: random.Random,
    ) -> Any:
        """Weighted random choice for operators."""
        total = sum(weights)
        if total <= 0:
            return rng.choice(items)

        r = rng.random() * total
        cumsum = 0.0
        for item, w in zip(items, weights):
            cumsum += w
            if r <= cumsum:
                return item
        return items[-1]

    def _build_adjacency(self) -> dict[str, dict[str, float]]:
        """Build adjacency dict from pheromone keys.

        Pheromone keys are "src->tgt" strings.  We parse them into
        {src: {tgt: tau, ...}, ...}.
        """
        adj: dict[str, dict[str, float]] = {}
        for key, tau in self.pheromones.items():
            if "->" not in key:
                continue
            src, tgt = key.split("->", 1)
            if src not in adj:
                adj[src] = {}
            adj[src][tgt] = tau
        return adj

    def _select_entry_role(
        self,
        adjacency: dict[str, dict[str, float]],
        rng: random.Random,
    ) -> str | None:
        """Select entry role: topology role with highest outgoing pheromone."""
        topo_roles = [
            r for r in adjacency
            if r.startswith("topo_")
        ]
        if not topo_roles:
            # Fallback: any role that has operators
            topo_roles = [
                r for r in self.operators_by_role if r.startswith("topo_")
            ]
        if not topo_roles:
            return None

        # Weight by sum of outgoing pheromones
        weights = []
        for role in topo_roles:
            out_tau = sum(adjacency.get(role, {}).values())
            weights.append(max(0.01, out_tau) ** self.alpha)

        return self._weighted_choice(topo_roles, weights, rng)

    def _select_next_role(
        self,
        current: str,
        adjacency: dict[str, dict[str, float]],
        rng: random.Random,
    ) -> str | None:
        """Select next role using pheromone-weighted softmax."""
        successors = adjacency.get(current, {})
        if not successors:
            return None

        roles = list(successors.keys())
        taus = [max(0.001, successors[r]) for r in roles]

        if self.temperature <= 0.01:
            # Greedy: argmax
            return roles[taus.index(max(taus))]

        # Softmax with temperature: P(i) ∝ exp(log(τ_i^α) / T)
        log_scores = [self.alpha * math.log(t) / self.temperature for t in taus]
        max_log = max(log_scores)
        exp_scores = [math.exp(s - max_log) for s in log_scores]

        return self._weighted_choice(roles, exp_scores, rng)

    def _weighted_choice(
        self,
        items: list[str],
        weights: list[float],
        rng: random.Random,
    ) -> str:
        """Weighted random choice."""
        total = sum(weights)
        if total <= 0:
            return rng.choice(items)

        r = rng.random() * total
        cumsum = 0.0
        for item, w in zip(items, weights):
            cumsum += w
            if r <= cumsum:
                return item
        return items[-1]

    def _validate_arch(self, arch: Any) -> bool:
        """Check that an architecture is valid.

        Uses pluggable ``arch_validator`` if set, otherwise falls back
        to Graph-specific validation (backwards compat).
        """
        if self.arch_validator is not None:
            return self.arch_validator(arch)

        # Default: GraphArchitecture validation
        return (
            hasattr(arch, "connectivity")
            and hasattr(arch, "operations")
            and len(arch.connectivity) == GRAPH_NUM_NODES
            and len(arch.operations) == GRAPH_NUM_NODES
            and all(0 <= c < GRAPH_MAX_CONN for c in arch.connectivity)
            and all(0 <= o < GRAPH_NUM_OPS for o in arch.operations)
        )


def load_snapshot_from_json(
    json_path: str,
    run_index: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load pheromone snapshot from a llm_sweep result JSON.

    Args:
        json_path: Path to nasbench_graph_llm_sweep_*.json.
        run_index: Which run to extract pheromones from (default: 0 = first).

    Returns:
        (pheromones_display, metadata) where pheromones_display is
        {"src->tgt": float, ...} and metadata has experiment info.
    """
    import json
    from pathlib import Path

    data = json.loads(Path(json_path).read_text())

    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {json_path}")

    if run_index >= len(runs):
        run_index = 0

    run = runs[run_index]
    pheromones = run.get("pheromones_display", {})

    if not pheromones:
        raise ValueError(
            f"No pheromones_display in run {run_index} of {json_path}"
        )

    metadata = {
        "source_file": str(json_path),
        "run_index": run_index,
        "llm": data.get("llm", "unknown"),
        "dataset": data.get("dataset", "unknown"),
        "best_accuracy": run.get("best_accuracy", 0.0),
        "n_evals": run.get("n_evals", 0),
        "total_tokens": data.get("total_tokens", 0),
    }

    return pheromones, metadata


def build_operators_by_role(
    operator_pool: Any | None = None,
) -> dict[str, list]:
    """Build the operator dispatch table from A₀ operators + optional L1 pool.

    If operator_pool is provided, compiles L1 operator code and mixes with A₀
    (same mechanism as run_nas_graph_benchmark.py uses for ACO).

    Args:
        operator_pool: Optional OperatorPool with L1 operators from LLM synthesis.

    Returns:
        {"role_name": [callable, ...], ...}
    """
    from src.geakg.generic_operators.graph_architecture import (
        GRAPH_ARCHITECTURE_OPERATORS,
    )
    from src.geakg.core.schemas.nas import NASRoleSchema

    schema = NASRoleSchema()
    operators_by_role: dict[str, list] = {}

    for role in schema.get_all_roles():
        ops = GRAPH_ARCHITECTURE_OPERATORS.get_operators(role)
        if ops:
            operators_by_role[role] = list(ops)

    if operator_pool is not None:
        compiled = _compile_operator_pool(operator_pool)
        n_l1 = 0
        for role, l1_ops in compiled.items():
            if role not in operators_by_role:
                operators_by_role[role] = []
            operators_by_role[role].extend(l1_ops)
            n_l1 += len(l1_ops)
        logger.info(
            f"[NASSymbolicExecutor] Loaded {n_l1} L1 operators "
            f"from pool (total: {sum(len(v) for v in operators_by_role.values())})"
        )

    return operators_by_role


def _compile_operator_pool(pool: Any) -> dict[str, list]:
    """Compile an OperatorPool's code strings into callable functions.

    Same mechanism as run_nas_graph_benchmark._compile_operator_pool().
    """
    compiled: dict[str, list] = {}

    for role in pool.roles:
        funcs = []
        for op in pool.get_operators_for_role(role):
            try:
                namespace: dict = {}
                exec(compile(op.code, f"<{op.name}>", "exec"), namespace)
                fn = None
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith("_"):
                        fn = obj
                        break
                if fn is not None:
                    funcs.append(fn)
            except Exception as e:
                logger.debug(f"[L1] Failed to compile {op.name}: {e}")

        if funcs:
            compiled[role] = funcs

    return compiled
